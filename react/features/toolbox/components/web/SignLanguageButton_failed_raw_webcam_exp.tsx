import React, { useState, useEffect, useRef, Component, ErrorInfo, PropsWithChildren } from 'react';
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
  confidence: number | null;
  error: string | null;
  isListenOnly: boolean;
  isSubtitlesCleared: boolean;
  isDeviceSupported: boolean;
}

const initialState: SignLanguageState = { 
  text: '', 
  confidence: null,
  error: null, 
  isListenOnly: false, 
  isSubtitlesCleared: false, 
  isDeviceSupported: true 
};

const SIGN_LANGUAGE_FEATURE = 'sign-language';

const signLanguageReducer = (state = initialState, action: { type: string; text?: string; confidence?: number; error?: string; isListenOnly?: boolean; isDeviceSupported?: boolean }) => {
  switch (action.type) {
    case 'UPDATE_SIGN_LANGUAGE_SUBTITLES':
      return { ...state, text: action.text || '', confidence: action.confidence || null, error: null, isSubtitlesCleared: false };
    case 'CLEAR_SIGN_LANGUAGE_SUBTITLES':
      return { ...state, text: '', confidence: null, error: null, isSubtitlesCleared: true };
    case 'SET_SIGN_LANGUAGE_ERROR':
      return { ...state, error: action.error || null, confidence: null, isSubtitlesCleared: false };
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

/**
 * Extracts keypoints from MediaPipe Hands results.
 * Returns a 126-element array: 63 for left hand, 63 for right hand.
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
      .map((landmark: any) => [landmark.x, landmark.y, landmark.z])
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
 * Centers keypoints around wrist midpoint.
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

const SignLanguageButton: React.FC<PropsWithChildren<ButtonProps>> = ({ t, 'aria-label': ariaLabel, className, isListenOnly, isDeviceSupported, dispatch }) => {
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
  const canvasElementRef = useRef<HTMLCanvasElement | null>(null);
  const popupWindowRef = useRef<Window | null>(null);
  const isCleanedUp = useRef<boolean>(false);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const lastStablePredictionRef = useRef<string | null>(null);
  const predictionStateRef = useRef<'LAST_SIGN' | 'BACKGROUND'>('BACKGROUND');

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

  // Helper function to clean up video element, canvas, and media stream
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
    if (canvasElementRef.current) {
      if (canvasElementRef.current.parentNode) {
        canvasElementRef.current.parentNode.removeChild(canvasElementRef.current);
      }
      canvasElementRef.current = null;
      console.log('Canvas element cleaned up');
    }
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach(track => track.stop());
      mediaStreamRef.current = null;
      console.log('Media stream stopped');
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
        console.log('MediaPipe Hands initialized');

        await tf.setBackend('webgl');
        console.log('TensorFlow.js backend set to webgl');

        const modelPath = `static/sign_language_model_tfjs_conv/model.json`;
        const loadedModel = await tf.loadLayersModel(modelPath);
        console.log('Model loaded');

        tf.tidy(() => {
          const dummyInput = tf.zeros([1, 30, 126]);
          loadedModel.predict(dummyInput).dispose();
          dummyInput.dispose();
        });
        console.log('Model warmed up');

        APP.conference._room.addCommandListener('sign_language', (data, participantId) => {
          const detectedSign = data.value;
          console.log(`Received sign from ${participantId}: ${detectedSign}`);
          const currentState = APP.store.getState();
          const isListenOnly = currentState['features/sign-language']?.isListenOnly || false;
          if (isListenOnly) {
            dispatch({
              type: 'UPDATE_SIGN_LANGUAGE_SUBTITLES',
              text: detectedSign,
              confidence: null,
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
      console.warn('Hands not initialized');
      return tf.zeros([1, 30, 126]);
    }

    return tf.tidy(() => {
      const keypoints = extractKeypoints(results);
      console.log('Keypoints shape:', keypoints.length);
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
  const predictSign = async (tensor: tf.Tensor): Promise<{ label: string; confidence: number }> => {
    if (!model || !labels.length) {
      console.warn('Model or labels not loaded');
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
      prediction.dispose();
      console.log(`Prediction: ${labels[labelIndex]}, Confidence: ${confidence}`);
      return { label: confidence > 0.95 ? labels[labelIndex] : 'BACKGROUND', confidence };
    } catch (error) {
      console.error('Prediction error:', error);
      return { label: 'Prediction failed', confidence: 0 };
    }
  };

  // Update subtitles with state machine logic
  const updateSubtitles = (text: string, confidence: number) => {
    if (isListenOnly) return;
    if (APP.store) {
      console.log(`Prediction: ${text}, Confidence: ${confidence}`);
      const isSign = labels.includes(text) && text !== 'BACKGROUND' && text !== 'SPACE' && text !== 'BACKSPACE';
      if (isSign && (!lastStablePredictionRef.current || lastStablePredictionRef.current === 'BACKGROUND' || text !== lastStablePredictionRef.current)) {
        dispatch({
          type: 'UPDATE_SIGN_LANGUAGE_SUBTITLES',
          text,
          confidence,
        });
        lastStablePredictionRef.current = text;
        predictionStateRef.current = 'LAST_SIGN';
        console.log(`Subtitles updated: ${text}`);
      } else if (text === 'BACKGROUND') {
        if (predictionStateRef.current === 'LAST_SIGN') {
          lastStablePredictionRef.current = 'BACKGROUND';
          predictionStateRef.current = 'BACKGROUND';
          console.log('Transition to BACKGROUND');
        }
      } else if (text === 'SPACE' || text === 'BACKSPACE') {
        dispatch({
          type: 'UPDATE_SIGN_LANGUAGE_SUBTITLES',
          text,
          confidence,
        });
        lastStablePredictionRef.current = text;
        console.log(`Special character: ${text}`);
      }
    }
  };

  // Process video frames
  const processVideoFrames = (): (() => void) => {
    let shouldContinue = true;
    let animationFrameId: number | null = null;

    const processFrame = async (timestamp: number) => {
      if (!shouldContinue || isCleanedUp.current) {
        console.log('Stopping frame processing');
        return;
      }
      if (timestamp - lastFrameTimeRef.current < 66) {
        animationFrameId = requestAnimationFrame(processFrame);
        return;
      }
      lastFrameTimeRef.current = timestamp;

      try {
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
    videoElement.autoplay = true;
    videoElementRef.current = videoElement;

    const canvasElement = document.createElement('canvas');
    canvasElement.width = 640;
    canvasElement.height = 480;
    canvasElementRef.current = canvasElement;
    const ctx = canvasElement.getContext('2d');

    navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480, facingMode: 'user' }
    }).then(stream => {
      mediaStreamRef.current = stream;
      videoElement.srcObject = stream;
      console.log('Webcam stream attached');
    }).catch(error => {
      console.error('Error accessing webcam:', error);
      dispatch({ type: 'SET_SIGN_LANGUAGE_ERROR', error: 'errors.webcamAccessFailed' });
    });

    videoElement.onloadedmetadata = () => {
      console.log('Video metadata loaded:', {
        width: videoElement.videoWidth,
        height: videoElement.videoHeight,
        readyState: videoElement.readyState,
      });
      videoElement.play().catch(error => {
        console.error('Failed to play video:', error);
        dispatch({ type: 'SET_SIGN_LANGUAGE_ERROR', error: 'errors.videoPlayFailed' });
      });
      const checkVideoReady = setInterval(() => {
        if (videoElement.readyState >= 2 && videoElement.videoWidth > 0 && videoElement.videoHeight > 0) {
          clearInterval(checkVideoReady);
          popupWindowRef.current = window.open('', 'SignLanguageDebug', 'width=660,height=500');
          if (popupWindowRef.current) {
            popupWindowRef.current.document.body.style.background = '#000';
            popupWindowRef.current.document.body.style.margin = '0';
            popupWindowRef.current.document.body.appendChild(canvasElement);
            popupWindowRef.current.document.title = 'Video Input (Flipped)';
            console.log('Popup window opened');
          }
        }
      }, 100);
    };

    const camera = new Camera(videoElement, {
      onFrame: async () => {
        if (videoElementRef.current && canvasElementRef.current && ctx && videoElementRef.current.readyState >= 2) {
          ctx.clearRect(0, 0, canvasElement.width, canvasElement.height);
          ctx.save();
          ctx.scale(-1, 1);
          ctx.translate(-640, 0);
          ctx.drawImage(videoElementRef.current, 0, 0, 640, 480);
          ctx.restore();
          console.log('Frame drawn on canvas');

          if (handsRef.current) {
            try {
              await handsRef.current.send({ image: videoElementRef.current });
              console.log('Frame sent to MediaPipe');
            } catch (err) {
              console.error('Error sending frame to MediaPipe:', err);
            }
          } else {
            console.warn('MediaPipe Hands not ready');
          }
        } else {
          console.warn('Video not ready:', {
            readyState: videoElementRef.current?.readyState,
            width: videoElementRef.current?.videoWidth,
            height: videoElementRef.current?.videoHeight,
          });
        }
      },
      width: 640,
      height: 480,
    });

    camera.start().then(() => {
      console.log('Camera started');
    }).catch(error => {
      console.error('Error starting camera:', error);
      dispatch({ type: 'SET_SIGN_LANGUAGE_ERROR', error: 'errors.cameraStartFailed' });
    });
    cameraRef.current = camera;

    handsRef.current?.onResults(async (results: HandResults) => {
      console.log('MediaPipe results received:', {
        multiHandLandmarks: !!results.multiHandLandmarks,
        multiHandedness: !!results.multiHandedness,
      });
      try {
        const tensor = await preprocessFrame(results);
        const { label: prediction, confidence } = await predictSign(tensor);
        predictionsRef.current.push(prediction);
        if (predictionsRef.current.length > 10) predictionsRef.current.shift();
        const isStable = predictionsRef.current.length === 10 &&
          predictionsRef.current.every(p => p === prediction);

        if (isStable && (prediction === 'BACKGROUND' || prediction === 'Unknown')) {
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
          updateSubtitles('BACKGROUND', 0);
        } else if (isStable && prediction !== 'BACKGROUND') {
          unknownStartTimeRef.current = null;
          APP.conference._room.sendCommand('sign_language', { value: prediction });
          updateSubtitles(prediction, confidence);
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
      lastStablePredictionRef.current = null;
      predictionStateRef.current = 'BACKGROUND';
    };
  };

  // Wait for conference
  const waitForConference = () => {
    return new Promise<void>((resolve, reject) => {
      if (APP.conference && APP.conference.isJoined()) {
        resolve();
      } else {
        dispatch({ type: 'SET_SIGN_LANGUAGE_ERROR', error: 'errors.conferenceNotJoined' });
        APP.conference.on(JitsiMeetJS.events.conference.CONFERENCE_JOINED, resolve);
      }
    });
  };

  // Handle translation toggle
  const handleTranslationToggle = async (enable: boolean) => {
    if (enable) {
      try {
        await waitForConference();
        const currentState = APP.store.getState();
        const isListenOnly = currentState['features/sign-language']?.isListenOnly || false;
        if (isListenOnly) {
          alert(t('signLanguage.disableListenOnlyFirst'));
          return;
        }
        animationFrameRef.current = processVideoFrames();
        setIsTranslationEnabled(true);
      } catch (error) {
        console.error('Error enabling translation:', error);
        dispatch({ type: 'SET_SIGN_LANGUAGE_ERROR', error: 'errors.conferenceNotJoined' });
      }
    } else {
      dispatch({ type: 'CLEAR_SIGN_LANGUAGE_SUBTITLES' });
      if (animationFrameRef.current) {
        animationFrameRef.current();
        animationFrameRef.current = null;
        console.log('Animation frame canceled in handleClick');
      }
      cleanupVideoElement();
      closePopupWindow();
      setIsTranslationEnabled(false);
      lastStablePredictionRef.current = null;
      predictionStateRef.current = 'BACKGROUND';
    }
  };

  const handleClick = () => {
    if (!isDeviceSupported) {
      dispatch({ type: 'SET_SIGN_LANGUAGE_ERROR', error: 'errors.deviceUnsupported' });
      return;
    }
    if (isListenOnly && !isTranslationEnabled) {
      alert(t('signLanguage.disableListenOnlyFirst'));
      return;
    }
    handleTranslationToggle(!isTranslationEnabled);
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

const ListenOnlySignLanguageButton: React.FC<PropsWithChildren<ListenOnlyButtonProps>> = ({ t, 'aria-label': ariaLabel, className, isListenOnly, isTranslationEnabled, isDeviceSupported, dispatch }) => {
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
  confidence: number | null;
  error: string | null;
  isListenOnly: boolean;
  isSubtitlesCleared: boolean;
}

const SignLanguageOverlay: React.FC<PropsWithChildren<OverlayProps>> = ({ subtitles, confidence, error, t, isListenOnly, isSubtitlesCleared }) => {
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
    if (subtitles.trim()) {
      if (subtitles === 'BACKSPACE') {
        setPredictions(prev => prev.slice(0, -1));
        console.log('Backspace detected, removed last character');
      } else if (subtitles === 'SPACE') {
        (async () => {
          try {
            const data = await postToGemini(predictions);
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
              setPredictions(predictions.slice(-50));
            }
          } catch (err) {
            console.error('Failed to fetch from Gemini:', err);
            setPredictions(predictions.slice(-50));
          }
        })();
      } else if (subtitles !== 'BACKGROUND') {
        setPredictions(prev => (prev + subtitles).slice(-50));
        console.log('Updated predictions:', prev + subtitles);
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
    confidence: state['features/sign-language']?.confidence || null,
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