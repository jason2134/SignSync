import React, { useState, useEffect, useRef, Component, ErrorInfo } from 'react';
import { translate } from '../../../base/i18n/functions';
import { withTranslation, WithTranslation } from 'react-i18next';
import { connect } from 'react-redux';
import * as tf from '@tensorflow/tfjs';
import { Hands, Results as HandResults } from '@mediapipe/hands';
import { Camera } from '@mediapipe/camera_utils';
import '@tensorflow/tfjs-backend-webgl';
import JitsiMeetJS from '../../../base/lib-jitsi-meet';
import ReducerRegistry from '../../../base/redux/ReducerRegistry';

// Define single-hand and two-hand signs
const SINGLE_HAND_SIGNS = ['C', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'SPACE'];
const TWO_HAND_SIGNS = ['A', 'B', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'BACKSPACE'];

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

// Extract keypoints with x-coordinate flipping
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
      .map((landmark: any) => [1 - landmark.x, landmark.y, landmark.z])
      .flat();

    if (keypoints.length !== LANDMARK_COUNT * COORDS_PER_LANDMARK) {
      console.warn(`Invalid keypoint length for hand ${idx}: ${keypoints.length}`);
      continue;
    }

    const handedness = handednessData.label;
    if (handedness === 'Right') {
      leftHand.splice(0, keypoints.length, ...keypoints);
    } else {
      rightHand.splice(0, keypoints.length, ...keypoints);
    }
  }

  return [...leftHand, ...rightHand] as KeypointFrame;
};

// Normalize keypoints
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
      (wristLeft[2] + wristRight[2]) / 2.0,
    ];

    const centeredFrame = reshapedFrame.map(landmark => [
      landmark[0] - origin[0],
      landmark[1] - origin[1],
      landmark[2] - origin[2],
    ]);

    normalizedSequence.push(centeredFrame.flat());
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
  const predictionsRef = useRef<number[]>([]);
  const sentenceRef = useRef<string[]>([]);
  const lastFrameTimeRef = useRef<number>(0);
  const unknownStartTimeRef = useRef<number | null>(null);
  const videoElementRef = useRef<HTMLVideoElement | null>(null);
  const isCleanedUp = useRef<boolean>(false);
  const backgroundCountRef = useRef<number>(0); // Track consecutive BACKGROUND predictions

  const THRESHOLD = 0.9;
  const BACKGROUND_DEBOUNCE_FRAMES = 45; // ~3 seconds at 15fps
  const TIMEOUT_DURATION = 12000; // Retain 12-second timeout

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

  const cleanupVideoElement = () => {
    if (videoElementRef.current) {
      videoElementRef.current.srcObject = null;
      videoElementRef.current.style.display = 'none';
      if (videoElementRef.current.parentNode) {
        videoElementRef.current.parentNode.removeChild(videoElementRef.current);
      }
      videoElementRef.current = null;
      console.log('Video element cleaned up');
    }
  };

  useEffect(() => {
    let isMounted = true;

    const initialize = async () => {
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
        if (!isMounted) return;
        handsRef.current = hands;

        await tf.setBackend('webgl');

        const modelPath = `static/tfjs_ann_model_converted_23_06/model.json`;
        const loadedModel = await tf.loadLayersModel(modelPath);

        tf.tidy(() => {
          const dummyInput = tf.zeros([1, 30 * 126]);
          loadedModel.predict(dummyInput).dispose();
          dummyInput.dispose();
        });

        APP.conference._room.addCommandListener('sign_language', (data, participantId) => {
          const detectedSign = data.value;
          console.log(`Received sign from ${participantId}: ${detectedSign}`);
          if (isListenOnly) {
            dispatch({
              type: 'UPDATE_SIGN_LANGUAGE_SUBTITLES',
              text: detectedSign,
            });
          }
        });

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
      }
      if (cameraRef.current) {
        cameraRef.current.stop();
        cameraRef.current = null;
      }
      if (handsRef.current) {
        handsRef.current.close().catch(err => console.error('Error closing hands:', err));
        handsRef.current = null;
      }
      if (model) {
        model.dispose();
        setModel(null);
      }
      cleanupVideoElement();
    };
  }, [dispatch]);

  const isValidSequence = (sequence: number[][], expectedSign?: string): boolean => {
    const requiresTwoHands = expectedSign
      ? TWO_HAND_SIGNS.includes(expectedSign)
      : predictionsRef.current.some(p => TWO_HAND_SIGNS.includes(labels[p]));
    let validFrameCount = 0;
    for (const frame of sequence) {
      const leftHand = frame.slice(0, 63);
      const rightHand = frame.slice(63, 126);
      const leftNonZero = leftHand.some(val => Math.abs(val) > 0.001);
      const rightNonZero = rightHand.some(val => Math.abs(val) > 0.001);
      if (requiresTwoHands ? leftNonZero && rightNonZero : leftNonZero || rightNonZero) {
        validFrameCount++;
      }
    }
    // Require at least 50% of frames to be valid to consider the sequence valid
    return validFrameCount >= sequence.length * 0.5;
  };

  const preprocessFrame = async (results: HandResults): Promise<tf.Tensor | null> => {
    if (!handsRef.current || !results.multiHandLandmarks || results.multiHandLandmarks.length === 0) {
      console.log('No hands detected, skipping prediction');
      return null;
    }

    const handCount = results.multiHandLandmarks.length;
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
        return tf.zeros([1, 30 * 126]);
      }

      if (handCount < 2 && predictionsRef.current.some(p => TWO_HAND_SIGNS.includes(labels[p]))) {
        console.log('Two-hand sign expected but only one hand detected, returning zero tensor');
        return tf.zeros([1, 30 * 126]);
      }

      const normalizedSequence = normalizeKeypoints(paddedSequence as KeypointSequence);
      const window = normalizedSequence.flat();
      return tf.tensor2d([window], [1, 30 * 126]);
    });

    const isZeroTensor = tf.equal(tensor, tf.zeros([1, 30 * 126])).all().dataSync()[0];
    if (isZeroTensor) {
      tensor.dispose();
      return null;
    }

    return tensor;
  };

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

      prediction.dispose();
      return { label: predictedLabel, confidence };
    } catch (error) {
      console.error('Prediction error:', error);
      return { label: 'Prediction failed', confidence: 0 };
    }
  };

  const updateSubtitles = (text: string) => {
    if (isListenOnly) return;
    if (APP.store) {
      dispatch({
        type: 'UPDATE_SIGN_LANGUAGE_SUBTITLES',
        text,
      });
    }
  };

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
      canvas.width = 640;
      canvas.height = 480;
      const ctx = canvas.getContext('2d');
      if (!ctx) throw new Error('No canvas context');
      ctx.drawImage(bitmap, 0, 0, 640, 480);
      return ctx.getImageData(0, 0, 640, 480);
    } catch (error) {
      throw error;
    }
  };

  const processVideoFrames = (videoTrack: any): (() => void) => {
    let shouldContinue = true;
    let animationFrameId: number | null = null;

    const processFrame = async (timestamp: number) => {
      if (!shouldContinue || !videoTrack || isCleanedUp.current) {
        console.log('Stopping frame processing');
        return;
      }
      if (timestamp - lastFrameTimeRef.current < 66) {
        animationFrameId = requestAnimationFrame(processFrame);
        return;
      }
      lastFrameTimeRef.current = timestamp;

      try {
        if (sequenceRef.current.length < 30) {
          animationFrameId = requestAnimationFrame(processFrame);
          return;
        }
        animationFrameId = requestAnimationFrame(processFrame);
      } catch (error) {
        console.error('Frame processing error:', error);
        dispatch({ type: 'SET_SIGN_LANGUAGE_ERROR', error: 'errors.frameProcessingFailed' });
      }
    };

    const videoElement = document.createElement('video');
    videoElement.width = 640;
    videoElement.height = 480;
    videoElement.muted = true;
    videoElementRef.current = videoElement;

    try {
      const mediaStream = videoTrack.stream || videoTrack.getStream?.();
      if (!(mediaStream instanceof MediaStream)) {
        throw new Error('No valid MediaStream available from videoTrack');
      }
      videoElement.srcObject = mediaStream;
      console.log('Video stream attached');
    } catch (error) {
      console.error('Error attaching video stream:', error);
      dispatch({ type: 'SET_SIGN_LANGUAGE_ERROR', error: 'errors.videoStreamFailed' });
      return () => {};
    }

    videoElement.onloadedmetadata = () => {
      videoElement.width = videoElement.videoWidth || 640;
      videoElement.height = videoElement.videoHeight || 480;
      videoElement.play().catch(error => {
        console.error('Failed to play video element:', error);
      });
    };

    const camera = new Camera(videoElement, {
      onFrame: async () => {
        if (
          handsRef.current &&
          videoElement.readyState >= 2 &&
          videoElement.videoWidth > 0 &&
          videoElement.videoHeight > 0
        ) {
          await handsRef.current.send({ image: videoElement }).catch(err => console.error('Error sending frame:', err));
        } else {
          console.warn('Video element not ready:', {
            readyState: videoElement.readyState,
            width: videoElement.videoWidth,
            height: videoElement.videoHeight,
          });
        }
      },
      width: 640,
      height: 480,
    });
    camera.start().catch(error => {
      console.error('Error starting camera:', error);
      dispatch({ type: 'SET_SIGN_LANGUAGE_ERROR', error: 'errors.cameraStartFailed' });
    });
    cameraRef.current = camera;

    handsRef.current?.onResults(async (results: HandResults) => {
      try {
        if (!results.multiHandLandmarks || results.multiHandLandmarks.length === 0) {
          console.log('No hands detected, pushing BACKGROUND prediction');
          predictionsRef.current.push(labels.indexOf('BACKGROUND'));
          if (predictionsRef.current.length > 10) predictionsRef.current.shift();
          backgroundCountRef.current += 1;
          if (backgroundCountRef.current >= BACKGROUND_DEBOUNCE_FRAMES) {
            if (unknownStartTimeRef.current === null) {
              unknownStartTimeRef.current = performance.now();
            }
            const elapsedTime = performance.now() - (unknownStartTimeRef.current || performance.now());
            if (elapsedTime >= TIMEOUT_DURATION) {
              dispatch({ type: 'CLEAR_SIGN_LANGUAGE_SUBTITLES' });
              unknownStartTimeRef.current = null;
              predictionsRef.current = [];
              sentenceRef.current = [];
              backgroundCountRef.current = 0;
            }
          }
          APP.conference._room.sendCommand('sign_language', { value: 'BACKGROUND' });
          updateSubtitles('');
          return;
        }

        const tensor = await preprocessFrame(results);
        if (!tensor) {
          console.log('No valid tensor, pushing BACKGROUND prediction');
          predictionsRef.current.push(labels.indexOf('BACKGROUND'));
          if (predictionsRef.current.length > 10) predictionsRef.current.shift();
          backgroundCountRef.current += 1;
          if (backgroundCountRef.current >= BACKGROUND_DEBOUNCE_FRAMES) {
            if (unknownStartTimeRef.current === null) {
              unknownStartTimeRef.current = performance.now();
            }
            const elapsedTime = performance.now() - (unknownStartTimeRef.current || performance.now());
            if (elapsedTime >= TIMEOUT_DURATION) {
              dispatch({ type: 'CLEAR_SIGN_LANGUAGE_SUBTITLES' });
              unknownStartTimeRef.current = null;
              predictionsRef.current = [];
              sentenceRef.current = [];
              backgroundCountRef.current = 0;
            }
          }
          APP.conference._room.sendCommand('sign_language', { value: 'BACKGROUND' });
          updateSubtitles('');
          return;
        }

        const prediction = await predictSign(tensor);
        predictionsRef.current.push(labels.indexOf(prediction.label));
        if (predictionsRef.current.length > 10) predictionsRef.current.shift();

        // Reset timeout and background count on valid sign detection
        if (prediction.label !== 'BACKGROUND' && prediction.confidence > THRESHOLD) {
          unknownStartTimeRef.current = null;
          backgroundCountRef.current = 0;
        }

        if (predictionsRef.current.length === 10 && prediction.confidence > THRESHOLD) {
          const mostCommonPrediction = predictionsRef.current.reduce((a, b, i, arr) =>
            arr.filter(x => x === a).length >= arr.filter(x => x === b).length ? a : b
          );
          if (predictionsRef.current.every(p => p === mostCommonPrediction)) {
            const predictedLabel = labels[mostCommonPrediction];
            if (predictedLabel !== 'BACKGROUND') {
              if (sentenceRef.current.length === 0 || sentenceRef.current[sentenceRef.current.length - 1] !== predictedLabel) {
                sentenceRef.current.push(predictedLabel);
                if (sentenceRef.current.length > 5) {
                  sentenceRef.current.shift();
                }
                updateSubtitles(predictedLabel);
                APP.conference._room.sendCommand('sign_language', { value: predictedLabel });
              }
            } else {
              backgroundCountRef.current += 1;
              if (backgroundCountRef.current >= BACKGROUND_DEBOUNCE_FRAMES) {
                if (unknownStartTimeRef.current === null) {
                  unknownStartTimeRef.current = performance.now();
                }
                const elapsedTime = performance.now() - (unknownStartTimeRef.current || performance.now());
                if (elapsedTime >= TIMEOUT_DURATION) {
                  dispatch({ type: 'CLEAR_SIGN_LANGUAGE_SUBTITLES' });
                  unknownStartTimeRef.current = null;
                  predictionsRef.current = [];
                  sentenceRef.current = [];
                  backgroundCountRef.current = 0;
                }
              }
              updateSubtitles('');
              APP.conference._room.sendCommand('sign_language', { value: 'BACKGROUND' });
            }
          }
        }

        tensor.dispose();
      } catch (error) {
        console.error('Error processing MediaPipe results:', error);
      }
    });

    requestAnimationFrame(processFrame);
    return () => {
      shouldContinue = false;
      isCleanedUp.current = true;
      if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
      }
      if (cameraRef.current) {
        cameraRef.current.stop();
        cameraRef.current = null;
      }
      cleanupVideoElement();
      sequenceRef.current = [];
      predictionsRef.current = [];
      sentenceRef.current = [];
      unknownStartTimeRef.current = null;
      backgroundCountRef.current = 0;
    };
  };

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
        }
        cleanupVideoElement();
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
          if (subtitles == 'SPACE' && newPredictions.length > 1) {
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