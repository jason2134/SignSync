import React, { useState, useEffect, useRef } from 'react';
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
}

const initialState: SignLanguageState = { text: '', error: null };

const SIGN_LANGUAGE_FEATURE = 'sign-language';

const signLanguageReducer = (state = initialState, action: { type: string; text?: string; error?: string }) => {
  switch (action.type) {
    case 'UPDATE_SIGN_LANGUAGE_SUBTITLES':
      return { ...state, text: action.text || '', error: null };
    case 'SET_SIGN_LANGUAGE_ERROR':
      return { ...state, error: action.error || null };
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
 * Defaults to right hand if handedness data is invalid.
 */
const extractKeypoints = (results: HandResults | null): KeypointFrame => {
  const leftHand = new Array(LANDMARK_COUNT * COORDS_PER_LANDMARK).fill(0);
  const rightHand = new Array(LANDMARK_COUNT * COORDS_PER_LANDMARK).fill(0);

  console.debug('MediaPipe results:', {
    multiHandLandmarks: results?.multiHandLandmarks?.length,
    multiHandedness: results?.multiHandedness?.length,
    rawHandedness: results?.multiHandedness ? JSON.stringify(results.multiHandedness, null, 2) : null,
  });

  if (!results || !results.multiHandLandmarks || !results.multiHandedness) {
    console.warn('No valid hand detection results, returning zeros');
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
      console.warn(`Invalid keypoints length for hand ${idx}:`, keypoints.length);
      continue;
    }

    if (!handednessData || !handednessData.label) {
      console.warn(`No valid handedness label for hand ${idx}, defaulting to right hand:`, handednessData);
      keypoints.forEach((val, i) => (rightHand[i] = val));
      continue;
    }

    const handedness = handednessData.label;
    if (handedness === 'Left') {
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

    // Process left hand (indices 0–62)
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

    // Process right hand (indices 63–125)
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
}

const SignLanguageButton: React.FC<ButtonProps> = ({ t, 'aria-label': ariaLabel, className }) => {
  const [isTranslationEnabled, setIsTranslationEnabled] = useState(false);
  const [isDeviceSupported, setIsDeviceSupported] = useState(true);
  const animationFrameRef = useRef<(() => void) | null>(null);
  const [model, setModel] = useState<tf.LayersModel | null>(null);
  const [labels, setLabels] = useState<string[]>([]);
  const handsRef = useRef<Hands | null>(null);
  const cameraRef = useRef<Camera | null>(null);
  const sequenceRef = useRef<number[][]>([]);
  const predictionsRef = useRef<string[]>([]);
  const lastFrameTimeRef = useRef<number>(0);

  // Check device compatibility
  useEffect(() => {
    const checkDevice = async () => {
      const webglVersion = await tf.env().get('WEBGL_VERSION');
      if (!webglVersion) {
        setIsDeviceSupported(false);
        APP.store.dispatch({ type: 'SET_SIGN_LANGUAGE_ERROR', error: 'errors.deviceUnsupported' });
      }
    };
    checkDevice();
  }, []);

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
          modelComplexity: 1,
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
          APP.store.dispatch({ type: 'SET_SIGN_LANGUAGE_ERROR', error: 'errors.initializationFailed' });
        }
      }
    };

    initialize();

    return () => {
      isMounted = false;
      if (animationFrameRef.current) {
        animationFrameRef.current();
        animationFrameRef.current = null;
      }
      if (cameraRef.current) {
        cameraRef.current.stop();
        cameraRef.current = null;
      }
      if (handsRef.current) {
        handsRef.current.close();
        handsRef.current = null;
      }
      if (model) {
        model.dispose();
        setModel(null);
      }
    };
  }, []);

  // Preprocess frame
  const preprocessFrame = async (results: HandResults): Promise<tf.Tensor> => {
    if (!handsRef.current) {
      console.warn('MediaPipe Hands not initialized');
      return tf.zeros([1, 30, 126]);
    }

    try {
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
    } catch (error) {
      console.error('Error preprocessing frame:', error);
      return tf.zeros([1, 30, 126]);
    }
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
      return confidence > 0.7 ? labels[labelIndex] : 'Unknown';
    } catch (error) {
      console.error('Prediction error:', error);
      return 'Prediction failed';
    }
  };

  // Update subtitles
  const updateSubtitles = (text: string) => {
    if (APP.store) {
      APP.store.dispatch({
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
    console.debug('Video track:', videoTrack);
    return videoTrack || null;
  };

  // Extract a single frame (used for initial validation)
  const extractFrame = async (videoTrack: any): Promise<ImageData> => {
    if (!videoTrack || !videoTrack.isVideoTrack() || videoTrack.isMuted() || videoTrack.videoType !== 'camera') {
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
      console.error('Failed to grab frame:', error);
      throw error;
    }
  };

  // Process video frames
  const processVideoFrames = (videoTrack: any): (() => void) => {
    let shouldContinue = true;
    let animationFrameId: number | null = null;

    const processFrame = async (timestamp: number) => {
      if (!shouldContinue || !videoTrack || videoTrack.isMuted()) {
        console.debug('Stopping frame processing: muted or invalid track');
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
        // Note: Actual frame processing is handled by Camera's onResults
        animationFrameId = requestAnimationFrame(processFrame);
      } catch (error) {
        console.error('Frame processing error:', error);
        APP.store.dispatch({ type: 'SET_SIGN_LANGUAGE_ERROR', error: 'errors.frameProcessingFailed' });
      }
    };

    // Start MediaPipe Camera
    const videoElement = document.createElement('video');
    try {
      // Use videoTrack.stream or videoTrack.getStream() to get MediaStream
      const mediaStream = videoTrack.stream || videoTrack.getStream?.();
      if (!(mediaStream instanceof MediaStream)) {
        throw new Error('No valid MediaStream available from videoTrack');
      }
      videoElement.srcObject = mediaStream;
    } catch (error) {
      console.error('Failed to set videoElement.srcObject:', error);
      APP.store.dispatch({ type: 'SET_SIGN_LANGUAGE_ERROR', error: 'errors.videoStreamFailed' });
      return () => {};
    }

    videoElement.onloadedmetadata = () => {
      console.debug('Video metadata:', videoElement.videoWidth, videoElement.videoHeight);
    };

    const camera = new Camera(videoElement, {
      onFrame: async () => {
        if (handsRef.current && videoElement) {
          await handsRef.current.send({ image: videoElement });
        }
      },
      width: 320,
      height: 240,
    });
    camera.start().catch(error => {
      console.error('Failed to start MediaPipe Camera:', error);
      APP.store.dispatch({ type: 'SET_SIGN_LANGUAGE_ERROR', error: 'errors.cameraStartFailed' });
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
        if (isStable && prediction !== 'Unknown') {
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
      if (animationFrameId) cancelAnimationFrame(animationFrameId);
      if (cameraRef.current) {
        cameraRef.current.stop();
        cameraRef.current = null;
      }
      sequenceRef.current = [];
      predictionsRef.current = [];
    };
  };

  // Wait for conference
  const waitForConference = () => {
    return new Promise<void>((resolve) => {
      if (APP.conference && APP.conference.isJoined()) {
        resolve();
      } else {
        APP.conference.on(JitsiMeetJS.events.conference.CONFERENCE_JOINED, resolve);
      }
    });
  };

  const handleClick = async () => {
    if (!isDeviceSupported) {
      APP.store.dispatch({ type: 'SET_SIGN_LANGUAGE_ERROR', error: 'errors.deviceUnsupported' });
      return;
    }
    setIsTranslationEnabled(prev => {
      const newValue = !prev;
      if (newValue) {
        waitForConference().then(() => {
          const attemptGetStream = async (attempts = 3, delay = 500): Promise<void> => {
            const videoTrack = getLocalVideoStream();
            if (videoTrack && !videoTrack.isMuted() && videoTrack.isVideoTrack()) {
              try {
                const frame = await extractFrame(videoTrack);
                animationFrameRef.current = processVideoFrames(videoTrack);
              } catch (error) {
                console.error('Error extracting frame:', error);
                APP.store.dispatch({ type: 'SET_SIGN_LANGUAGE_ERROR', error: 'errors.initializationFailed' });
              }
            } else if (attempts > 0) {
              await new Promise(resolve => setTimeout(resolve, delay));
              return attemptGetStream(attempts - 1, delay);
            } else {
              APP.store.dispatch({ type: 'SET_SIGN_LANGUAGE_ERROR', error: 'errors.initializationFailed' });
            }
          };
          attemptGetStream();
        });
      } else {
        if (animationFrameRef.current) {
          animationFrameRef.current();
          animationFrameRef.current = null;
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

const ConnectedSignLanguageButton = translate(connect()(SignLanguageButton));

// SignLanguageOverlay component
interface OverlayProps extends WithTranslation {
  subtitles: string;
  error: string | null;
}

const SignLanguageOverlay: React.FC<OverlayProps> = ({ subtitles, error, t }) => {
  const [predictions, setPredictions] = useState<string[]>([]);

  useEffect(() => {
    if (error) {
      setPredictions([t(error)]);
      return;
    }
    if (subtitles.trim()) {
      setPredictions(prev => [...prev, subtitles]);
    }
  }, [subtitles, error, t]);

  useEffect(() => {
    if (!APP.store) return;
    const unsubscribe = APP.store.subscribe(() => {});
    return () => unsubscribe();
  }, []);

  const handleClear = () => {
    setPredictions([]);
  };

  if (!predictions.length) return null;

  return (
    <>
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
            zIndex: 99999;
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
      <div className="sign-language-overlay" aria-live="polite">
        <span className="prediction-text">
          {predictions.join(' ')}
        </span>
        {predictions.length > 0 && (
          <button className="clear-button" onClick={handleClear}>
            {t('signLanguage.clear')}
          </button>
        )}
      </div>
    </>
  );
};

const ConnectedSignLanguageOverlay = withTranslation()(connect(
  (state: any) => ({
    subtitles: state['features/sign-language']?.text || '',
    error: state['features/sign-language']?.error || null,
  })
)(SignLanguageOverlay));

const SignLanguageApp: React.FC = () => {
  useEffect(() => {
    if (!APP.store) return;
    if (process.env.NODE_ENV === 'development') {
      APP.store.dispatch({ type: 'UPDATE_SIGN_LANGUAGE_SUBTITLES', text: 'TEST' });
    }
  }, []);
  return <ConnectedSignLanguageButton />;
};

const SignLanguageOverlayApp: React.FC = () => {
  return <ConnectedSignLanguageOverlay />;
};

export { SignLanguageApp, SignLanguageOverlayApp, ConnectedSignLanguageButton };
export default SignLanguageApp;