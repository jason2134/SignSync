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
    return [...leftHand, ...rightHand] as KeypointFrame;
  }

  const handCount = Math.min(results.multiHandLandmarks.length, results.multiHandedness.length);
  for (let idx = 0; idx < handCount; idx++) {
    const landmarks = results.multiHandLandmarks[idx];
    const handednessData = results.multiHandedness[idx];

    const keypoints = landmarks
      .map((landmark: any) => [1 - landmark.x, landmark.y, landmark.z])
      .flat();

    if (keypoints.length !== LANDMARK_COUNT * COORDS_PER_LANDMARK) {
      continue;
    }

    if (!handednessData || !handednessData.label) {
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

  return [...leftHand, ...rightHand] as KeypointFrame;
};

/**
 * Normalizes a sequence of keypoint frames.
 * Centers each hand by wrist (landmark 0), scales by middle finger MCP (landmark 9).
 * Returns zeros for undetected hands or invalid scaling.
 */
const normalizeKeypoints = (sequence: KeypointSequence): KeypointSequence => {
  const normalizedSequence: KeypointSequence = [];

  for (const frame of sequence) {
    if (frame.length !== KEYPOINTS_PER_FRAME) {
      throw new Error(`Invalid frame length: expected ${KEYPOINTS_PER_FRAME}, got ${frame.length}`);
    }

    const normalizedFrame = new Array(KEYPOINTS_PER_FRAME).fill(0) as KeypointFrame;

    // Process left hand (indices 0–62, user's left hand)
    const leftHand = frame.slice(0, LANDMARK_COUNT * COORDS_PER_LANDMARK);
    let hasLeftHand = leftHand.some(val => val !== 0);

    if (hasLeftHand) {
      const wristX = frame[WRIST_INDEX * COORDS_PER_LANDMARK];
      const wristY = frame[WRIST_INDEX * COORDS_PER_LANDMARK + 1];
      const wristZ = frame[WRIST_INDEX * COORDS_PER_LANDMARK + 2];

      const centeredLeft = new Array(LANDMARK_COUNT * COORDS_PER_LANDMARK).fill(0);
      for (let i = 0; i < LANDMARK_COUNT; i++) {
        const base = i * COORDS_PER_LANDMARK;
        centeredLeft[base] = frame[base] - wristX;
        centeredLeft[base + 1] = frame[base + 1] - wristY;
        centeredLeft[base + 2] = frame[base + 2] - wristZ;
      }

      const mcpBase = MIDDLE_MCP_INDEX * COORDS_PER_LANDMARK;
      const handSize = Math.sqrt(
        centeredLeft[mcpBase] ** 2 +
        centeredLeft[mcpBase + 1] ** 2 +
        centeredLeft[mcpBase + 2] ** 2
      );

      if (handSize > 0) {
        for (let i = 0; i < LANDMARK_COUNT * COORDS_PER_LANDMARK; i++) {
          normalizedFrame[i] = centeredLeft[i] / handSize;
        }
      }
    }

    // Process right hand (indices 63–125, user's right hand)
    const rightHand = frame.slice(LANDMARK_COUNT * COORDS_PER_LANDMARK);
    let hasRightHand = rightHand.some(val => val !== 0);

    if (hasRightHand) {
      const wristBase = LANDMARK_COUNT * COORDS_PER_LANDMARK;
      const wristX = frame[wristBase];
      const wristY = frame[wristBase + 1];
      const wristZ = frame[wristBase + 2];

      const centeredRight = new Array(LANDMARK_COUNT * COORDS_PER_LANDMARK).fill(0);
      for (let i = 0; i < LANDMARK_COUNT; i++) {
        const base = i * COORDS_PER_LANDMARK + wristBase;
        const outBase = i * COORDS_PER_LANDMARK;
        centeredRight[outBase] = frame[base] - wristX;
        centeredRight[outBase + 1] = frame[base + 1] - wristY;
        centeredRight[outBase + 2] = frame[base + 2] - wristZ;
      }

      const mcpBase = MIDDLE_MCP_INDEX * COORDS_PER_LANDMARK;
      const handSize = Math.sqrt(
        centeredRight[mcpBase] ** 2 +
        centeredRight[mcpBase + 1] ** 2 +
        centeredRight[mcpBase + 2] ** 2
      );

      if (handSize > 0) {
        for (let i = 0; i < LANDMARK_COUNT * COORDS_PER_LANDMARK; i++) {
          normalizedFrame[i + wristBase] = centeredRight[i] / handSize;
        }
      }
    }

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
  const predictionsRef = useRef<string[]>([]);
  const lastFrameTimeRef = useRef<number>(0);
  const unknownStartTimeRef = useRef<number | null>(null);
  const videoElementRef = useRef<HTMLVideoElement | null>(null);
  const popupWindowRef = useRef<Window | null>(null);
  const isCleanedUp = useRef<boolean>(false);

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

  // Helper function to clean up video element
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

  // Helper function to close popup window
  const closePopupWindow = () => {
    if (popupWindowRef.current && !popupWindowRef.current.closed) {
      try {
        popupWindowRef.current.close();
        console.log('Popup window closed');
      } catch (error) {
        console.error('Error closing popup window:', error);
      }
      popupWindowRef.current = null;
    }
  };

  // Initialize MediaPipe Hands and TensorFlow model
  useEffect(() => {
    let isMounted = true;

    const initialize = async () => {
      try {
        // Initialize MediaPipe Hands
        const hands = new Hands({
          locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.4.1646424915/${file}`,
        });
        hands.setOptions({
          maxNumHands: 2,
          modelComplexity: 1, //1
          minDetectionConfidence: 0.5,
          minTrackingConfidence: 0.5,
        });
        await hands.initialize();
        if (!isMounted) return;
        handsRef.current = hands;

        // Set TensorFlow backend
        await tf.setBackend('webgl');

        // Load model
        const modelPath = `static/sign_language_model_tfjs_conv/model.json`;
        const loadedModel = await tf.loadLayersModel(modelPath);

        // Warm up model
        tf.tidy(() => {
          const dummyInput = tf.zeros([1, 30, 126]);
          loadedModel.predict(dummyInput).dispose();
          dummyInput.dispose();
        });

        // Init listener for sign language
        APP.conference._room.addCommandListener('sign_language', (data, participantId) => {
          const detectedSign = data.value;
          console.log(`Received sign from ${participantId}: ${detectedSign}`);
          const currentState = APP.store.getState();
          const isListenOnly = currentState['features/sign-language']?.isListenOnly || false;
          console.log("ListenOnly State", isListenOnly);
          if (isListenOnly) {
            console.log("Dispatching sign");
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
          'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']);
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
      cleanupVideoElement();
      closePopupWindow();
    };
  }, [dispatch]);

  // Preprocess frame
  const preprocessFrame = async (results: HandResults): Promise<tf.Tensor> => {
    if (!handsRef.current) {
      return tf.zeros([1, 30, 126]);
    }

    return tf.tidy(() => {
      const keypoints = extractKeypoints(results);
      sequenceRef.current.push(keypoints);
      if (sequenceRef.current.length > 30) {
        sequenceRef.current.shift();
      }
      const paddedSequence = sequenceRef.current.length < 30
        ? [...new Array(30 - sequenceRef.current.length).fill(new Array(126).fill(0)), ...sequenceRef.current]
        : sequenceRef.current;
      const normalizedSequence = normalizeKeypoints(paddedSequence as KeypointSequence);
      return tf.tensor3d([normalizedSequence], [1, 30, 126]);
    });
  };

  // Predict sign
  const predictSign = async (tensor: tf.Tensor): Promise<string> => {
    if (!model || !labels.length) return 'Model or labels not loaded';
    try {
      const prediction = model.predict(tensor) as tf.Tensor;
      const [probs, labelIndexTensor] = await Promise.all([
        prediction.data(),
        prediction.argMax(-1).data(),
      ]);
      const labelIndex = labelIndexTensor[0];
      const confidence = probs[labelIndex];
      prediction.dispose();
      return confidence > 0.85 ? labels[labelIndex] : 'Unknown';
    } catch (error) {
      console.error('Prediction error:', error);
      return 'Prediction failed';
    }
  };

  // Update subtitles
  const updateSubtitles = (text: string) => {
    if (isListenOnly) return; // Prevent local predictions from updating subtitles when listen-only is active
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

  // Process video frames
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
      const checkVideoReady = setInterval(() => {
        if (videoElement.readyState >= 2 && videoElement.videoWidth > 0 && videoElement.videoHeight > 0) {
          clearInterval(checkVideoReady);
          popupWindowRef.current = window.open('', 'SignLanguageDebug', 'width=660,height=500');
          if (popupWindowRef.current) {
            popupWindowRef.current.document.body.style.background = '#000';
            popupWindowRef.current.document.body.style.margin = '0';
            popupWindowRef.current.document.body.appendChild(videoElement);
            popupWindowRef.current.document.title = 'Video Input (Raw)';
            console.log('Popup window opened');
          }
        }
      }, 100);
    };

    const camera = new Camera(videoElement, {
      onFrame: async () => {
        if (handsRef.current && videoElement.readyState >= 2 && videoElement.videoWidth > 0 && videoElement.videoHeight > 0) {
          await handsRef.current.send({ image: videoElement }).catch(err => console.error('Error sending frame:', err));
        }else {
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
        const tensor = await preprocessFrame(results);
        const prediction = await predictSign(tensor);
        predictionsRef.current.push(prediction);
        if (predictionsRef.current.length > 10) predictionsRef.current.shift();
        const isStable = predictionsRef.current.length === 10 &&
          predictionsRef.current.every(p => p === prediction);

        // Handle Unknown prediction timing
        if (isStable && prediction === 'Unknown') {
          if (!unknownStartTimeRef.current) {
            unknownStartTimeRef.current = performance.now();
          } else {
            const elapsedTime = performance.now() - unknownStartTimeRef.current;
            if (elapsedTime >= 8000) { // 8 seconds
              dispatch({ type: 'CLEAR_SIGN_LANGUAGE_SUBTITLES' });
              unknownStartTimeRef.current = null; // Reset timer
              predictionsRef.current = []; // Clear predictions
            }
          }
          APP.conference._room.sendCommand('sign_language', { value: 'Unknown' });
          updateSubtitles('');
        } else if (isStable && prediction !== 'Unknown') {
          unknownStartTimeRef.current = null; // Reset timer on valid detection
          APP.conference._room.sendCommand('sign_language', { value: prediction });
          updateSubtitles(prediction);
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
        console.log('Animation frame canceled in processVideoFrames');
      }
      if (cameraRef.current) {
        cameraRef.current.stop();
        cameraRef.current = null;
        console.log('Camera stopped in processVideoFrames');
      }
      cleanupVideoElement();
      closePopupWindow();
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
      alert(t('signLanguage.disableListenOnlyFirst')); // Show popup message
      return;
    }
    setIsTranslationEnabled(prev => {
      const newValue = !prev;
      if (newValue) {
        waitForConference().then(() => {
          const currentState = APP.store.getState();
          const isListenOnly = currentState['features/sign-language']?.isListenOnly || false;
          if (isListenOnly) {
            alert(t('signLanguage.disableListenOnlyFirst')); // Show popup message
            setIsTranslationEnabled(false); // Revert state change
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
        // Clear subtitles when disabling Sign Language
        dispatch({ type: 'CLEAR_SIGN_LANGUAGE_SUBTITLES' });
        if (animationFrameRef.current) {
          animationFrameRef.current();
          animationFrameRef.current = null;
          console.log('Animation frame canceled in handleClick');
        }
        cleanupVideoElement();
        closePopupWindow();
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
      alert(t('Disable the Sign Language Button to turn on Listen Only Mode')); // Show popup message
      return;
    }
    const newListenOnlyState = !isListenOnly;
    dispatch({ type: 'TOGGLE_LISTEN_ONLY', isListenOnly: newListenOnlyState });
    if (newListenOnlyState) {
      // Clear subtitles when enabling Listen Only
      dispatch({ type: 'CLEAR_SIGN_LANGUAGE_SUBTITLES' });
    } else {
      // Clear subtitles when disabling Listen Only
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
  const [predictions, setPredictions] = useState<string>(''); // Change to string instead of string[]

  useEffect(() => {
    if (error) {
      setPredictions(t(error));
      return;
    }
    if (isSubtitlesCleared) {
      setPredictions('');
      return;
    }
    if (subtitles.trim() && subtitles !== 'Unknown') {
      setPredictions(prev => {
        const newPredictions = prev + subtitles; // Append new sign to existing string
        if (newPredictions.length > 1) {
          // Perform async operation
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
                setPredictions(upperResString.slice(-50)); // Limit to last 50 chars to prevent overflow
              } else {
                console.error('Invalid response structure from postToGemini:', data);
                setPredictions(newPredictions.slice(-50)); // Fallback to current predictions
              }
            } catch (err) {
              console.error('Failed to fetch from Gemini:', err);
              setPredictions(newPredictions.slice(-50)); // Fallback
            }
          })();
          return newPredictions.slice(-50); // Return current predictions while async runs
        }

        return newPredictions.slice(-50); // Limit length
      });
    }
  }, [subtitles, error, t, isSubtitlesCleared]);

  // Gemini backend POST function
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