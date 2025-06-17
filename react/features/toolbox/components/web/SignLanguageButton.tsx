
import React, { useState, useEffect, useRef, Component, ErrorInfo } from 'react';
import { translate } from '../../../base/i18n/functions';
import { withTranslation, WithTranslation } from 'react-i18next';
import { connect } from 'react-redux';
import * as tf from '@tensorflow/tfjs';
import { Hands, Results as HandResults, HAND_CONNECTIONS, NormalizedLandmarkList } from '@mediapipe/hands';
import { drawConnectors, drawLandmarks } from '@mediapipe/drawing_utils';
import { Camera } from '@mediapipe/camera_utils';
import '@tensorflow/tfjs-backend-webgl';
import JitsiMeetJS from '../../../base/lib-jitsi-meet';
import ReducerRegistry from '../../../base/redux/ReducerRegistry';

// Custom DrawingOptions interface to resolve TypeScript error
interface DrawingOptions {
  color?: string;
  fillColor?: string;
  lineWidth?: number;
  radius?: number;
}

// Define single-hand and two-hand signs
const SINGLE_HAND_SIGNS = ['C', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10'];
const TWO_HAND_SIGNS = ['A', 'B', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'];

// Redux reducer for sign language subtitles
interface SignLanguageState {
  text: string;
  error: string | null;
  isListenOnly: boolean;
  isSubtitlesCleared: boolean;
  isDeviceSupported: boolean;
}

const initialState: SignLanguageState = { 
  text: '', 
  error: null, 
  isListenOnly: false, 
  isSubtitlesCleared: false, 
  isDeviceSupported: true 
};

const SIGN_LANGUAGE_FEATURE = 'sign-language';

const signLanguageReducer = (state = initialState, action: { type: string; text?: string; error?: string; isListenOnly?: boolean; isDeviceSupported?: boolean }) => {
  switch (action.type) {
    case 'UPDATE_SIGN_LANGUAGE_SUBTITLES':
      return { ...state, text: action.text || '', error: null, isSubtitlesCleared: false };
    case 'CLEAR_SIGN_LANGUAGE_SUBTITLES':
      return { ...state, text: '', error: null, isSubtitlesCleared: true };
    case 'SET_SIGN_LANGUAGE_ERROR':
      return { ...state, error: action.error || null, isSubtitlesCleared: false };
    case 'TOGGLE_LISTEN_ONLY':
      console.log('Redux: Toggling listen-only:', action.isListenOnly);
      return { ...state, isListenOnly: action.isListenOnly ?? !state.isListenOnly, isSubtitlesCleared: false };
    case 'SET_DEVICE_SUPPORT':
      return { ...state, isDeviceSupported: action.isDeviceSupported ?? true };
    default:
      return state;
  }
};

ReducerRegistry.register(`features/${SIGN_LANGUAGE_FEATURE}`, signLanguageReducer);

// Type definitions
type KeypointFrame = number[] & { length: 126 };
type KeypointSequence = KeypointFrame[];

const LANDMARK_COUNT = 21;
const COORDS_PER_LANDMARK = 3;
const KEYPOINTS_PER_FRAME = LANDMARK_COUNT * COORDS_PER_LANDMARK * 2; // 126
const WRIST_INDEX = 0;
const MIDDLE_MCP_INDEX = 9;

/**
 * Extracts keypoints from MediaPipe Hands results.
 * Returns a 126-element array: 63 for left hand (21 landmarks × 3 coords), 63 for right hand.
 * Flips x-coordinates to match mirrored training data (user's left hand on left side).
 */
const extractKeypoints = (results: HandResults | null): KeypointFrame => {
  const leftHand = new Array(LANDMARK_COUNT * COORDS_PER_LANDMARK).fill(0);
  const rightHand = new Array(LANDMARK_COUNT * COORDS_PER_LANDMARK).fill(0);

  if (!results || !results.multiHandLandmarks || !results.multiHandedness) {
    console.log('No hands detected, returning zero keypoints');
    return [...leftHand, ...rightHand] as KeypointFrame;
  }

  const handCount = Math.min(results.multiHandLandmarks.length, results.multiHandedness.length);
  console.log(`Detected ${handCount} hand(s)`);
  for (let idx = 0; idx < handCount; idx++) {
    const landmarks = results.multiHandLandmarks[idx];
    const handednessData = results.multiHandedness[idx];

    const keypoints = landmarks
      .map((landmark: any) => [1 - landmark.x, landmark.y, landmark.z]) // Flip x for mirrored training data
      .flat();

    if (keypoints.length !== LANDMARK_COUNT * COORDS_PER_LANDMARK) {
      console.warn(`Invalid keypoint length for hand ${idx}: ${keypoints.length}`);
      continue;
    }

    if (!handednessData || !handednessData.label) {
      console.log('No handedness data, assigning to right hand');
      keypoints.forEach((val, i) => (rightHand[i] = val));
      continue;
    }

    const handedness = handednessData.label;
    if (handedness === 'Right') {
      keypoints.forEach((val, i) => (leftHand[i] = val));
    } else {
      keypoints.forEach((val, i) => (rightHand[i] = val));
    }
  }

  if (handCount === 1) {
    console.log('Single hand detected:', {
      leftHand: leftHand.some(val => Math.abs(val) > 0.001) ? leftHand : 'zeros',
      rightHand: rightHand.some(val => Math.abs(val) > 0.001) ? rightHand : 'zeros',
    });
  }

  return [...leftHand, ...rightHand] as KeypointFrame;
};

/**
 * Normalizes a sequence of keypoint frames.
 * Centers each hand by wrist (landmark 0), scales by middle finger MCP (landmark 9).
 * Returns zeros for undetected hands or invalid scaling.
 */
const normalizeKeypoints = (sequence: number[][]): number[][] => {
  const expectedLength = 126;
  if (!Array.isArray(sequence) || sequence.some(frame => frame.length !== expectedLength)) {
    throw new Error(`Invalid sequence: each frame must have ${expectedLength} elements`);
  }

  const normalizedSequence: number[][] = [];

  for (const frame of sequence) {
    const reshapedFrame: number[][] = [];
    for (let i = 0; i < 42; i++) {
      const base = i * 3;
      reshapedFrame.push([frame[base], frame[base + 1], frame[base + 2]]);
    }

    const wristLeft = reshapedFrame[0];
    const wristRight = reshapedFrame[21];
    const origin = [
      (wristLeft[0] + wristRight[0]) / 2.0,
      (wristLeft[1] + wristRight[1]) / 2.0,
      (wristLeft[2] + wristRight[2]) / 2.0
    ];

    const centeredFrame = reshapedFrame.map(landmark => [
      landmark[0] - origin[0],
      landmark[1] - origin[1],
      landmark[2] - origin[2]
    ]);

    const normalizedFrame: number[] = centeredFrame.flat();
    normalizedSequence.push(normalizedFrame);
  }

  return normalizedSequence;
};

// SignLanguageButton component
interface ButtonProps extends WithTranslation {
  'aria-label'?: string;
  className?: string;
  isListenOnly: boolean;
  isDeviceSupported: boolean;
  dispatch: (action: any) => void;
}

const SignLanguageButton: React.FC<ButtonProps> = ({ t, 'aria-label': ariaLabel, className, isListenOnly, isDeviceSupported, dispatch }) => {
  const [isTranslationEnabled, setIsTranslationEnabled] = useState(false);
  const animationFrameRef = useRef<(() => void) | null>(null);
  const [model, setModel] = useState<tf.LayersModel | null>(null);
  const [labels, setLabels] = useState<string[]>([]);
  const handsRef = useRef<Hands | null>(null);
  const cameraRef = useRef<Camera | null>(null);
  const sequenceRef = useRef<number[][]>([]);
  const predictionsRef = useRef<{ label: string; confidence: number }[]>([]);
  const lastFrameTimeRef = useRef<number>(0);
  const unknownStartTimeRef = useRef<number | null>(null);
  const videoElementRef = useRef<HTMLVideoElement | null>(null);
  const canvasElementRef = useRef<HTMLCanvasElement | null>(null);
  const isCleanedUp = useRef<boolean>(false);
  const isProcessingFrame = useRef<boolean>(false);
  const wasmErrorCount = useRef<number>(0);

  // Initialize MediaPipe Hands
  const initializeHands = async (): Promise<Hands | null> => {
    try {
      const hands = new Hands({
        locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.4.1646424915/${file}`,
      });
      hands.setOptions({
        maxNumHands: 2,
        modelComplexity: 1,
        minDetectionConfidence: 0.8,
        minTrackingConfidence: 0.8,
      });
      await hands.initialize();
      console.log('Hands initialized successfully');
      return hands;
    } catch (error) {
      console.error('Hands initialization failed:', error);
      dispatch({ type: 'SET_SIGN_LANGUAGE_ERROR', error: 'errors.initializationFailed' });
      return null;
    }
  };

  // Check device compatibility
  useEffect(() => {
    const checkDevice = async () => {
      try {
        const webglVersion = await tf.env().get('WEBGL_VERSION');
        const canvas = document.createElement('canvas');
        const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
        if (!webglVersion || !gl) {
          dispatch({ type: 'SET_DEVICE_SUPPORT', isDeviceSupported: false });
          dispatch({ type: 'SET_SIGN_LANGUAGE_ERROR', error: 'errors.deviceUnsupported' });
        } else {
          dispatch({ type: 'SET_DEVICE_SUPPORT', isDeviceSupported: true });
        }
      } catch (error) {
        console.error('Device check failed:', error);
        dispatch({ type: 'SET_DEVICE_SUPPORT', isDeviceSupported: false });
      }
    };
    checkDevice();
  }, [dispatch]);

  // Initialize MediaPipe Hands and TensorFlow model
  useEffect(() => {
    let isMounted = true;

    const initialize = async () => {
      try {
        const hands = await initializeHands();
        if (!isMounted || !hands) return;
        handsRef.current = hands;

        await tf.setBackend('webgl');

        const modelPath = `static/sign_language_model_tfjs_conv/model.json`;
        const loadedModel = await tf.loadLayersModel(modelPath);

        tf.tidy(() => {
          const dummyInput = tf.zeros([1, 30, 126]);
          loadedModel.predict(dummyInput).dispose();
          dummyInput.dispose();
        });

        APP.conference._room.addCommandListener('sign_language', (data, participantId) => {
          const detectedSign = data.value;
          console.log(`Received sign from ${participantId}: ${detectedSign}`);
          const currentState = APP.store.getState();
          const isListenOnly = currentState['features/sign-language']?.isListenOnly || false;
          if (isListenOnly) {
            dispatch({ type: 'UPDATE_SIGN_LANGUAGE_SUBTITLES', text: detectedSign });
          }
        });

        const canvasElement = document.createElement('canvas');
        canvasElement.style.position = 'absolute';
        canvasElement.style.top = '0';
        canvasElement.style.left = '0';
        canvasElement.style.zIndex = '1000';
        canvasElement.style.pointerEvents = 'none';
        canvasElement.width = 320;
        canvasElement.height = 240;

        const findVideoAndAppendCanvas = () => {
          const videoElements = document.querySelectorAll('video');
          for (const video of videoElements) {
            if (video.srcObject instanceof MediaStream || video.readyState >= 2) {
              const parent = video.parentElement;
              if (parent) {
                parent.appendChild(canvasElement);
                canvasElementRef.current = canvasElement;
                canvasElement.width = video.videoWidth || 320;
                canvasElement.height = video.videoHeight || 240;
                console.log('Canvas attached to video parent:', parent);
                break;
              }
            }
          }
          if (!canvasElementRef.current) {
            console.warn('No active video element found, retrying...');
            setTimeout(findVideoAndAppendCanvas, 500);
          }
        };

        findVideoAndAppendCanvas();

        const observer = new MutationObserver((mutations, obs) => {
          const videoFound = document.querySelector('video[srcObject], video[readyState]');
          if (videoFound && !canvasElementRef.current) {
            findVideoAndAppendCanvas();
            obs.disconnect();
          }
        });
        observer.observe(document.body, { childList: true, subtree: true });

        if (!isMounted) {
          loadedModel.dispose();
          return;
        }
        setModel(loadedModel);
        setLabels(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
          'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
          '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
          'SPACE', 'BACKSPACE', 'BACKGROUND']);
      } catch (error) {
        console.error('Initialization failed:', error);
        if (isMounted) {
          dispatch({ type: 'SET_SIGN_LANGUAGE_ERROR', error: 'errors.initializationFailed' });
        }
      }
    };

    initialize();

    return () => {
      isMounted = false;
      isCleanedUp.current = true;
      if (animationFrameRef.current) {
        animationFrameRef.current();
        animationFrameRef.current = null;
        console.log('Animation frame canceled');
      }
      if (cameraRef.current) {
        cameraRef.current.stop();
        cameraRef.current = null;
        console.log('Camera stopped');
      }
      if (handsRef.current) {
        handsRef.current.close().catch(err => console.error('Error closing hands:', err));
        handsRef.current = null;
        console.log('Hands closed');
      }
      if (model) {
        model.dispose();
        setModel(null);
        console.log('Model disposed');
      }
      if (canvasElementRef.current) {
        if (canvasElementRef.current.parentNode) {
          canvasElementRef.current.parentNode.removeChild(canvasElementRef.current);
        }
        canvasElementRef.current = null;
        console.log('Canvas element cleaned up');
      }
    };
  }, [dispatch]);

  // Validate sequence to ensure it contains meaningful data
  const isValidSequence = (sequence: number[][], expectedSign?: string): boolean => {
    const requiresTwoHands = expectedSign
      ? TWO_HAND_SIGNS.includes(expectedSign)
      : predictionsRef.current.some(p => TWO_HAND_SIGNS.includes(p.label));

    return sequence.some(frame => {
      const leftHand = frame.slice(0, 63);
      const rightHand = frame.slice(63, 126);
      const leftNonZero = leftHand.some(val => Math.abs(val) > 0.001);
      const rightNonZero = rightHand.some(val => Math.abs(val) > 0.001);
      return requiresTwoHands ? leftNonZero && rightNonZero : leftNonZero || rightNonZero;
    });
  };

  // Preprocess frame
  const preprocessFrame = async (results: HandResults): Promise<tf.Tensor | null> => {
    if (!handsRef.current || !results.multiHandLandmarks || results.multiHandLandmarks.length === 0) {
      console.log('No hands detected, skipping prediction');
      return null;
    }

    const tensor = tf.tidy(() => {
      const keypoints = extractKeypoints(results);
      sequenceRef.current.push(keypoints);
      if (sequenceRef.current.length > 30) {
        sequenceRef.current.shift();
      }
      const paddedSequence = sequenceRef.current.length < 30
        ? [...new Array(30 - sequenceRef.current.length).fill(new Array(126).fill(0)), ...sequenceRef.current]
        : sequenceRef.current;

      if (!isValidSequence(paddedSequence)) {
        console.log('Invalid sequence, returning zero tensor');
        return tf.zeros([1, 30, 126]);
      }

      if (results.multiHandLandmarks.length < 2 && predictionsRef.current.some(p => TWO_HAND_SIGNS.includes(p.label))) {
        console.log('Two-hand sign expected but only one hand detected, returning zero tensor');
        return tf.zeros([1, 30, 126]);
      }

      const normalizedSequence = normalizeKeypoints(paddedSequence as KeypointSequence);
      return tf.tensor3d([normalizedSequence], [1, 30, 126]);
    });

    const isZeroTensor = tf.equal(tensor, tf.zeros([1, 30, 126])).all().dataSync()[0];
    if (isZeroTensor) {
      tensor.dispose();
      return null;
    }

    return tensor;
  };

  // Predict sign with majority voting
  const predictSign = async (tensor: tf.Tensor): Promise<{ label: string; confidence: number }> => {
    if (!model || !labels.length) {
      console.error('Model or labels not loaded');
      return { label: 'Model or labels not loaded', confidence: 0 };
    }
    try {
      const prediction = model.predict(tensor) as tf.Tensor;
      const [probs, labelIndexTensor] = await Promise.all([
        prediction.data(),
        prediction.argMax(-1).data(),
      ]);
      const labelIndex = labelIndexTensor[0];
      const confidence = probs[labelIndex];
      const predictedLabel = labels[labelIndex];

      if (TWO_HAND_SIGNS.includes(predictedLabel)) {
        const sequenceValidForTwoHands = isValidSequence(sequenceRef.current, predictedLabel);
        if (!sequenceValidForTwoHands) {
          console.log(`Predicted two-hand sign ${predictedLabel} but sequence lacks both hands; defaulting to BACKGROUND`);
          prediction.dispose();
          return { label: 'BACKGROUND', confidence: 1.0 };
        }
      }

      console.log(`Prediction: ${predictedLabel}, Confidence: ${confidence}`);
      prediction.dispose();
      return { label: confidence > 0.95 ? predictedLabel : 'BACKGROUND', confidence };
    } catch (error) {
      console.error('Prediction error:', error);
      return { label: 'Prediction failed', confidence: 0 };
    }
  };

  // Update subtitles
  const updateSubtitles = (text: string) => {
    if (isListenOnly) return;
    if (APP.store) {
      dispatch({
        type: 'UPDATE_SIGN_LANGUAGE_SUBTITLES',
        text,
      });
    }
  };

  // Get local video stream
  const getLocalVideoStream = (): any | null => {
    const conference = APP.conference._room;
    if (!conference) return null;
    const localTracks = conference.getLocalTracks();
    const videoTrack = localTracks.find(track => track.getType() === 'video');
    if (videoTrack && videoTrack.isVideoTrack()) {
      const mediaStreamTrack = videoTrack.getTrack();
      if (mediaStreamTrack && mediaStreamTrack.readyState === 'live') {
        const settings = mediaStreamTrack.getSettings();
        console.log(`Video track resolution: ${settings.width}x${settings.height}`);
      } else {
        console.warn('Video track is not live or unavailable');
      }
    }
    return videoTrack || null;
  };

  // Extract a single frame
  const extractFrame = async (videoTrack: any): Promise<ImageData> => {
    if (!videoTrack || !videoTrack.isVideoTrack() || videoTrack.videoType !== 'camera') {
      throw new Error('Invalid video track');
    }
    const track = videoTrack.getTrack();
    if (!track || track.readyState !== 'live') {
      throw new Error('No valid MediaStreamTrack');
    }
    const imageCapture = new ImageCapture(track);
    try {
      const bitmap = await imageCapture.grabFrame();
      const canvas = document.createElement('canvas');
      canvas.width = 320;
      canvas.height = 240;
      const ctx = canvas.getContext('2d');
      if (!ctx) throw new Error('No canvas context');
      ctx.drawImage(bitmap, 0, 0, 320, 240);
      return ctx.getImageData(0, 0, 320, 240);
    } catch (error) {
      throw error;
    }
  };

  // Draw landmarks on canvas using MediaPipe draw_utils, with x-flip for mirror mode
  const drawHandLandmarks = (ctx: CanvasRenderingContext2D, results: HandResults) => {
    ctx.save(); // Save canvas state
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
    ctx.globalCompositeOperation = 'source-over';

    if (results.multiHandLandmarks && results.multiHandedness) {
      for (let idx = 0; idx < results.multiHandLandmarks.length; idx++) {
        const landmarks: NormalizedLandmarkList = results.multiHandLandmarks[idx];
        const handedness = results.multiHandedness[idx].label;

        // Flip x-coordinates to match mirrored video feed
        const mirroredLandmarks = landmarks.map(landmark => ({
          ...landmark,
          x: 1 - landmark.x,
        }));

        const landmarkStyles: DrawingOptions = {
          color: '#00FF00FF', // Green for right, red for left (RGBA)
          fillColor: '#00FF00FF',
          radius: 5,
        };
        const connectionStyles: DrawingOptions = {
          color: '#00FF00FF',
          lineWidth: 2,
        };

        drawConnectors(ctx, mirroredLandmarks, HAND_CONNECTIONS, connectionStyles);
        drawLandmarks(ctx, mirroredLandmarks, landmarkStyles);
      }
    }
    ctx.restore(); // Restore canvas state
  };

  // Process video frames with enhanced throttling
  const processVideoFrames = (videoTrack: any): (() => void) => {
    let shouldContinue = true;
    let animationFrameId: number | null = null;

    const processFrame = async (timestamp: number) => {
      if (!shouldContinue || !videoTrack || isCleanedUp.current || isProcessingFrame.current) {
        animationFrameId = requestAnimationFrame(processFrame);
        return;
      }
      if (timestamp - lastFrameTimeRef.current < 125) { // ~8 FPS
        animationFrameId = requestAnimationFrame(processFrame);
        return;
      }
      lastFrameTimeRef.current = timestamp;
      isProcessingFrame.current = true;

      try {
        if (sequenceRef.current.length < 30) {
          animationFrameId = requestAnimationFrame(processFrame);
          isProcessingFrame.current = false;
          return;
        }

        const mediaStreamTrack = videoTrack.getTrack();
        if (mediaStreamTrack && mediaStreamTrack.readyState === 'live') {
          const settings = mediaStreamTrack.getSettings();
          const canvas = canvasElementRef.current;
          if (canvas && (canvas.width !== settings.width || canvas.height !== settings.height)) {
            canvas.width = settings.width || 320;
            canvas.height = settings.height || 240;
            console.log(`Updated canvas resolution: ${settings.width}x${settings.height}`);
          }
        } else {
          console.warn('Video track not live, skipping frame');
          animationFrameId = requestAnimationFrame(processFrame);
          isProcessingFrame.current = false;
          return;
        }

        if (canvasElementRef.current) {
          const ctx = canvasElementRef.current.getContext('2d');
          if (ctx) {
            const videoElement = videoElementRef.current;
            if (videoElement && handsRef.current && videoElement.readyState >= 2) {
              await handsRef.current.send({ image: videoElement }).catch(err => {
                console.error('Error sending frame:', err, {
                  videoReadyState: videoElement.readyState,
                  videoDimensions: `${videoElement.videoWidth}x${videoElement.videoHeight}`,
                });
                if (err.message.includes('Aborted')) {
                  wasmErrorCount.current += 1;
                  if (wasmErrorCount.current >= 3) {
                    console.warn('Multiple WASM errors detected, reinitializing Hands...');
                    handsRef.current?.close().catch(e => console.error('Error closing Hands:', e));
                    initializeHands().then(newHands => {
                      if (newHands) {
                        handsRef.current = newHands;
                        wasmErrorCount.current = 0;
                        console.log('Hands reinitialized successfully');
                      }
                    });
                  }
                }
                dispatch({ type: 'SET_SIGN_LANGUAGE_ERROR', error: 'errors.frameProcessingFailed' });
              });
            } else {
              console.warn('Video element not ready or handsRef is null', {
                videoReadyState: videoElement?.readyState,
                handsRefExists: !!handsRef.current,
              });
            }
          }
        }

        animationFrameId = requestAnimationFrame(processFrame);
      } catch (error) {
        console.error('Frame processing error:', error);
        dispatch({ type: 'SET_SIGN_LANGUAGE_ERROR', error: 'errors.frameProcessingFailed' });
      } finally {
        isProcessingFrame.current = false;
      }
    };

    const videoElement = document.createElement('video');
    videoElement.muted = true;
    videoElementRef.current = videoElement;

    try {
      const mediaStream = videoTrack.stream || videoTrack.getStream?.();
      if (!(mediaStream instanceof MediaStream) || !mediaStream.active) {
        throw new Error('No valid or active MediaStream available from videoTrack');
      }
      videoElement.srcObject = mediaStream;
      videoElement.play().catch(err => console.error('Failed to play video:', err));
    } catch (error) {
      console.error('Error attaching video stream:', error);
      dispatch({ type: 'SET_SIGN_LANGUAGE_ERROR', error: 'errors.videoStreamFailed' });
      return () => {};
    }

    videoElement.onloadedmetadata = () => {
      console.log('Video element ready:', videoElement.readyState);
      if (canvasElementRef.current) {
        const mediaStreamTrack = videoTrack.getTrack();
        if (mediaStreamTrack && mediaStreamTrack.readyState === 'live') {
          const settings = mediaStreamTrack.getSettings();
          canvasElementRef.current.width = settings.width || 320;
          canvasElementRef.current.height = settings.height || 240;
          console.log(`Video track resolution: ${settings.width}x${settings.height}`);
        }
      }
    };

    videoElement.onerror = () => {
      console.error('Video element error');
      dispatch({ type: 'SET_SIGN_LANGUAGE_ERROR', error: 'errors.videoStreamFailed' });
    };

    const camera = new Camera(videoElement, {
      onFrame: async () => {
        const videoElement = videoElementRef.current;
        if (handsRef.current && videoElement && videoElement.readyState >= 2 && !isProcessingFrame.current) {
          await handsRef.current.send({ image: videoElement }).catch(err => console.error('Error sending frame:', err));
        }
      },
      width: 320,
      height: 240,
    });
    camera.start().catch(err => {
      console.error('Error starting camera:', err);
      dispatch({ type: 'SET_SIGN_LANGUAGE_ERROR', error: 'errors.cameraStartFailed' });
    });
    cameraRef.current = camera;

    handsRef.current?.onResults(async (results: HandResults) => {
      try {
        console.log('onResults called:', {
          multiHandLandmarks: results.multiHandLandmarks?.length,
          multiHandedness: results.multiHandedness?.length,
          timestamp: performance.now(),
          videoResolution: videoElementRef.current ? `${videoElementRef.current.videoWidth}x${videoElementRef.current.videoHeight}` : 'unknown',
        });

        if (canvasElementRef.current) {
          const ctx = canvasElementRef.current.getContext('2d');
          if (ctx) {
            drawHandLandmarks(ctx, results);
          }
        }

        if (!results.multiHandLandmarks || results.multiHandLandmarks.length === 0) {
          console.log('No hands detected, pushing BACKGROUND prediction');
          predictionsRef.current.push({ label: 'BACKGROUND', confidence: 1.0 });
          if (predictionsRef.current.length > 5) predictionsRef.current.shift(); // Reduced window size
          if (unknownStartTimeRef.current === null) {
            unknownStartTimeRef.current = performance.now();
          }
          const elapsedTime = performance.now() - (unknownStartTimeRef.current || performance.now());
          if (elapsedTime >= 8000) {
            dispatch({ type: 'CLEAR_SIGN_LANGUAGE_SUBTITLES' });
            unknownStartTimeRef.current = null;
            predictionsRef.current = [];
          }
          APP.conference._room.sendCommand('sign_language', { value: 'BACKGROUND' });
          updateSubtitles('');
          return;
        }

        if (results.multiHandLandmarks.length === 1) {
          console.log('Single hand detected, checking for single-hand signs');
        }

        const tensor = await preprocessFrame(results);
        if (!tensor) {
          console.log('No valid tensor, pushing BACKGROUND prediction');
          predictionsRef.current.push({ label: 'BACKGROUND', confidence: 1.0 });
          if (predictionsRef.current.length > 5) predictionsRef.current.shift();
          APP.conference._room.sendCommand('sign_language', { value: 'BACKGROUND' });
          updateSubtitles('');
          return;
        }

        const prediction = await predictSign(tensor);
        predictionsRef.current.push(prediction);
        if (predictionsRef.current.length > 5) predictionsRef.current.shift();

        console.log('Prediction result:', prediction);

        const predictionCounts: { [key: string]: { count: number; totalConfidence: number } } = {};
        predictionsRef.current.forEach(p => {
          if (!predictionCounts[p.label]) {
            predictionCounts[p.label] = { count: 0, totalConfidence: 0 };
          }
          predictionCounts[p.label].count += 1;
          predictionCounts[p.label].totalConfidence += p.confidence;
        });

        const mostCommon = Object.entries(predictionCounts).reduce((a, b) =>
          a[1].count > b[1].count || (a[1].count === b[1].count && a[1].totalConfidence > b[1].totalConfidence) ? a : b
        );
        const stablePrediction = mostCommon[0];
        const avgConfidence = mostCommon[1].totalConfidence / mostCommon[1].count;

        const isStable = mostCommon[1].count >= 4 && avgConfidence > 0.95; // Relaxed stability criteria

        if (isStable && (stablePrediction === 'BACKGROUND' || stablePrediction === 'Unknown')) {
          if (!unknownStartTimeRef.current) {
            unknownStartTimeRef.current = performance.now();
          } else {
            const elapsedTime = performance.now() - unknownStartTimeRef.current;
            if (elapsedTime >= 8000) {
              dispatch({ type: 'CLEAR_SIGN_LANGUAGE_SUBTITLES' });
              unknownStartTimeRef.current = null;
              predictionsRef.current = [];
            }
          }
          APP.conference._room.sendCommand('sign_language', { value: 'BACKGROUND' });
          updateSubtitles('');
        } else if (isStable && stablePrediction !== 'BACKGROUND') {
          unknownStartTimeRef.current = null;
          if (TWO_HAND_SIGNS.includes(stablePrediction) && results.multiHandLandmarks.length < 2) {
            console.log(`Stable prediction ${stablePrediction} is a two-hand sign but only ${results.multiHandLandmarks.length} hand(s) detected; defaulting to BACKGROUND`);
            APP.conference._room.sendCommand('sign_language', { value: 'BACKGROUND' });
            updateSubtitles('');
          } else {
            APP.conference._room.sendCommand('sign_language', { value: stablePrediction });
            updateSubtitles(stablePrediction);
          }
        }

        tensor.dispose();
      } catch (error) {
        console.error('Error processing MediaPipe results:', error, {
          multiHandLandmarks: results.multiHandLandmarks?.length,
          multiHandedness: results.multiHandedness?.length,
        });
        if (error.message.includes('Aborted')) {
          wasmErrorCount.current += 1;
          if (wasmErrorCount.current >= 3) {
            console.warn('Multiple WASM errors detected, reinitializing Hands...');
            handsRef.current?.close().catch(e => console.error('Error closing Hands:', e));
            initializeHands().then(newHands => {
              if (newHands) {
                handsRef.current = newHands;
                wasmErrorCount.current = 0;
                console.log('Hands reinitialized successfully');
              }
            });
          }
        }
      }
    });

    requestAnimationFrame(processFrame);
    return () => {
      shouldContinue = false;
      isCleanedUp.current = true;
      if (animationFrameId) cancelAnimationFrame(animationFrameId);
      if (cameraRef.current) {
        cameraRef.current.stop();
        cameraRef.current = null;
      }
      if (videoElementRef.current) {
        videoElementRef.current.srcObject = null;
        videoElementRef.current = null;
      }
      sequenceRef.current = [];
      predictionsRef.current = [];
      unknownStartTimeRef.current = null;
    };
  };

  // Wait for conference
  const waitForConference = () => {
    return new Promise<void>((resolve) => {
      if (APP.conference && APP.conference.isJoined()) {
        resolve();
      } else {
        dispatch({ type: 'SET_SIGN_LANGUAGE_ERROR', error: 'errors.conferenceNotJoined' });
        APP.conference.on(JitsiMeetJS.events.conference.CONFERENCE_JOINED, resolve);
      }
    });
  };

  const handleClick = async () => {
    if (!isDeviceSupported) {
      dispatch({ type: 'SET_SIGN_LANGUAGE_ERROR', error: 'errors.deviceUnsupported' });
      return;
    }
    if (isListenOnly && !isTranslationEnabled) {
      alert(t('signLanguage.disableListenOnlyFirst'));
      return;
    }
    setIsTranslationEnabled(prev => {
      const newValue = !prev;
      if (newValue) {
        waitForConference().then(() => {
          const currentState = APP.store.getState();
          const isListenOnly = currentState['features/sign-language']?.isListenOnly || false;
          if (isListenOnly) {
            alert(t('signLanguage.disableListenOnlyFirst'));
            setIsTranslationEnabled(false);
            return;
          }
          const attemptGetStream = async (attempts = 3, delay = 500): Promise<void> => {
            const videoTrack = getLocalVideoStream();
            if (videoTrack && videoTrack.isVideoTrack()) {
              try {
                const frame = await extractFrame(videoTrack);
                animationFrameRef.current = processVideoFrames(videoTrack);
              } catch (error) {
                console.error('Error processing video track:', error);
                dispatch({ type: 'SET_SIGN_LANGUAGE_ERROR', error: 'errors.initializationFailed' });
              }
            } else if (attempts > 0) {
              await new Promise(resolve => setTimeout(resolve, delay));
              return attemptGetStream(attempts - 1, delay);
            } else {
              console.error('Failed to get video stream after retries');
              dispatch({ type: 'SET_SIGN_LANGUAGE_ERROR', error: 'errors.initializationFailed' });
            }
          };
          attemptGetStream();
        });
      } else {
        dispatch({ type: 'CLEAR_SIGN_LANGUAGE_SUBTITLES' });
        if (animationFrameRef.current) {
          animationFrameRef.current();
          animationFrameRef.current = null;
          console.log('Animation frame canceled in handleClick');
        }
      }
      return newValue;
    });
  };

  return (
    <button
      className={className || 'toolbox-button'}
      aria-label={ariaLabel || t('toolbar.signLanguage')}
      onClick={handleClick}
      style={{ color: isTranslationEnabled ? '#00ff00' : '#ffffff' }}
      disabled={!isDeviceSupported}
    >
      {t('toolbar.signLanguage')}
    </button>
  );
};

const ConnectedSignLanguageButton = translate(connect(
  (state: any) => ({
    isListenOnly: state['features/sign-language']?.isListenOnly || false,
    isDeviceSupported: state['features/sign-language']?.isDeviceSupported || true,
  }),
  (dispatch) => ({ dispatch })
)(SignLanguageButton));

// ListenOnlySignLanguageButton Component
interface ListenOnlyButtonProps extends WithTranslation {
  'aria-label'?: string;
  className?: string;
  isListenOnly: boolean;
  isTranslationEnabled: boolean;
  isDeviceSupported: boolean;
  dispatch: (action: any) => void;
}

const ListenOnlySignLanguageButton: React.FC<ListenOnlyButtonProps> = ({ t, 'aria-label': ariaLabel, className, isListenOnly, isTranslationEnabled, isDeviceSupported, dispatch }) => {
  const handleClick = () => {
    if (isTranslationEnabled && !isListenOnly) {
      alert(t('Disable the Sign Language Button to turn on Listen Only Mode'));
      return;
    }
    const newListenOnlyState = !isListenOnly;
    dispatch({ type: 'TOGGLE_LISTEN_ONLY', isListenOnly: newListenOnlyState });
    if (newListenOnlyState) {
      dispatch({ type: 'CLEAR_SIGN_LANGUAGE_SUBTITLES' });
    } else {
      dispatch({ type: 'CLEAR_SIGN_LANGUAGE_SUBTITLES' });
    }
  };

  return (
    <button
      className={className || 'toolbox-button'}
      aria-label={ariaLabel || t('toolbar.listenOnlySignLanguage')}
      onClick={handleClick}
      style={{ color: isListenOnly ? '#00ff00' : '#ffffff' }}
      disabled={!isDeviceSupported}
    >
      {t('toolbar.listenOnlySignLanguage')}
    </button>
  );
};

const ConnectedListenOnlySignLanguageButton = translate(connect(
  (state: any) => ({
    isListenOnly: state['features/sign-language']?.isListenOnly || false,
    isTranslationEnabled: !!state['features/sign-language']?.text,
    isDeviceSupported: state['features/sign-language']?.isDeviceSupported || true,
  }),
  (dispatch) => ({ dispatch })
)(ListenOnlySignLanguageButton));

// SignLanguageOverlay component
interface OverlayProps extends WithTranslation {
  subtitles: string;
  error: string | null;
  isListenOnly: boolean;
  isSubtitlesCleared: boolean;
}

const SignLanguageOverlay: React.FC<OverlayProps> = ({ subtitles, error, t, isListenOnly, isSubtitlesCleared }) => {
  const [predictions, setPredictions] = useState<string>('');

  useEffect(() => {
    if (error) {
      setPredictions(t(error));
      return;
    }
    if (isSubtitlesCleared) {
      setPredictions('');
      return;
    }
    if (subtitles.trim() && subtitles !== 'BACKGROUND') {
      if (subtitles === 'BACKSPACE') {
        setPredictions(prev => prev.slice(0, -1));
      } else if (subtitles === 'SPACE') {
        setPredictions(prev => {
          const newPredictions = prev;
          if (subtitles === 'SPACE' && newPredictions.length > 1) {
            (async () => {
              try {
                const data = await postToGemini(newPredictions);
                if (
                  data &&
                  data.candidates &&
                  Array.isArray(data.candidates) &&
                  data.candidates[0] &&
                  data.candidates[0].content &&
                  data.candidates[0].content.parts &&
                  Array.isArray(data.candidates[0].content.parts) &&
                  data.candidates[0].content.parts[0] &&
                  typeof data.candidates[0].content.parts[0].text === 'string'
                ) {
                  const resString = data.candidates[0].content.parts[0].text.trim();
                  console.log('Gemini Response:', resString);
                  const upperResString = resString.toUpperCase();
                  setPredictions(upperResString.slice(-50));
                } else {
                  console.error('Invalid response structure from postToGemini:', data);
                  setPredictions(newPredictions.slice(-50));
                }
              } catch (err) {
                console.error('Failed to fetch from Gemini:', err);
                setPredictions(newPredictions.slice(-50));
              }
            })();
            return newPredictions.slice(-50);
          }
          return newPredictions.slice(-50);
        });
      } else {
        setPredictions(prev => {
          const newPredictions = prev + subtitles;
          console.log('Updated predictions:', newPredictions);
          return newPredictions.slice(-50);
        });
      }
    }
  }, [subtitles, error, t, isSubtitlesCleared]);

  const postToGemini = async (text: string) => {
    const prompt = text.replace(/\s/g, '');
    console.log('Request prompt:', prompt);
    try {
      const response = await fetch('/ask', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ prompt: prompt }),
      });
      const data = await response.json();
      console.log('Gemini Response:', data);
      return data;
    } catch (err) {
      console.error('Failed to post to Gemini:', err);
      return null;
    }
  };

  useEffect(() => {
    if (!APP.store) return;
    const unsubscribe = APP.store.subscribe(() => {});
    return () => unsubscribe();
  }, []);

  const handleClear = () => {
    setPredictions('');
    APP.store.dispatch({ type: 'CLEAR_SIGN_LANGUAGE_SUBTITLES' });
  };

  if (!predictions) return null;

  return (
    <div className="sign-language-overlay" aria-live="polite">
      <style>
        {`
          @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
          }
          .sign-language-overlay {
            position: fixed;
            bottom: 3rem;
            left: 50%;
            transform: translateX(-50%);
            z-index: 99999;
            max-width: 80vw;
            background-color: rgba(0, 0, 0, 0.7);
            border-radius: 0.5rem;
            padding: 0.5rem 1rem;
            animation: fadeIn 0.3s ease-in forwards;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
            white-space: nowrap;
            overflow-x: auto;
          }
          .prediction-text {
            color: white;
            font-size: 1.5rem;
            font-weight: bold;
            display: inline;
            margin-right: 1rem;
          }
          .clear-button {
            background-color: #ff4444;
            color: white;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 0.3rem;
            cursor: pointer;
            font-size: 1rem;
            margin-left: 1rem;
          }
          .clear-button:hover {
            background-color: #cc0000;
          }
        `}
      </style>
      <span className="prediction-text">{predictions}</span>
      {predictions.length > 0 && (
        <button className="clear-button" onClick={handleClear}>
          {t('Translation Clear')}
        </button>
      )}
    </div>
  );
};

const ConnectedSignLanguageOverlay = withTranslation()(connect(
  (state: any) => ({
    subtitles: state['features/sign-language']?.text || '',
    error: state['features/sign-language']?.error || null,
    isListenOnly: state['features/sign-language']?.isListenOnly || false,
    isSubtitlesCleared: state['features/sign-language']?.isSubtitlesCleared || false,
  })
)(SignLanguageOverlay));

// Error Boundary Component
interface ErrorBoundaryProps {
  children: React.ReactNode;
}

interface ErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
}

class SignLanguageErrorBoundary extends Component<ErrorBoundaryProps & WithTranslation, ErrorBoundaryState> {
  state: ErrorBoundaryState = {
    hasError: false,
    error: null,
  };

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('SignLanguageErrorBoundary caught error:', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div style={{ color: 'red', padding: '1rem' }}>
          <h2>{this.props.t('signLanguage.error')}</h2>
          <p>{this.state.error?.message || this.props.t('signLanguage.unexpectedError')}</p>
        </div>
      );
    }

    return this.props.children;
  }
}

const TranslatedSignLanguageErrorBoundary = withTranslation()(SignLanguageErrorBoundary);

const SignLanguageApp: React.FC = () => {
  useEffect(() => {
    if (!APP.store) return;
    if (process.env.NODE_ENV === 'development') {
      APP.store.dispatch({ type: 'UPDATE_SIGN_LANGUAGE_SUBTITLES', text: 'TEST' });
    }
  }, []);
  return (
    <TranslatedSignLanguageErrorBoundary>
      <ConnectedSignLanguageButton />
      <ConnectedListenOnlySignLanguageButton />
      <ConnectedSignLanguageOverlay />
    </TranslatedSignLanguageErrorBoundary>
  );
};

const SignLanguageOverlayApp: React.FC = () => {
  return (
    <TranslatedSignLanguageErrorBoundary>
      <ConnectedSignLanguageOverlay />
    </TranslatedSignLanguageErrorBoundary>
  );
};

export { SignLanguageApp, SignLanguageOverlayApp, ConnectedSignLanguageButton, ConnectedListenOnlySignLanguageButton };
export default SignLanguageApp;