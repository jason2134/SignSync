import React, { useState, useEffect, useRef } from 'react';
import { translate } from '../../../base/i18n/functions';
import { withTranslation, WithTranslation } from 'react-i18next';
import { connect, Provider } from 'react-redux';
import * as tf from '@tensorflow/tfjs';
import { Holistic } from '@mediapipe/holistic';
import '@tensorflow/tfjs-backend-webgl';
import JitsiMeetJS from '../../../base/lib-jitsi-meet';
import ReducerRegistry from '../../../base/redux/ReducerRegistry';
import PersistenceRegistry from '../../../base/redux/PersistenceRegistry';

// Redux reducer for sign language subtitles
interface SignLanguageState {
  text: string;
  error: string | null;
}

const initialState: SignLanguageState = { text: '', error: null };

const SIGN_LANGUAGE_FEATURE = 'sign-language';

const signLanguageReducer = (state = initialState, action: { type: string; text?: string; error?: string }) => {
  console.log('signLanguageReducer called with action:', action);
  switch (action.type) {
    case 'UPDATE_SIGN_LANGUAGE_SUBTITLES':
      console.log('Updating sign language state to:', { text: action.text || '' });
      return { ...state, text: action.text || '', error: null };
    case 'SET_SIGN_LANGUAGE_ERROR':
      console.log('Setting sign language error:', { error: action.error || null });
      return { ...state, error: action.error || null };
    default:
      return state;
  }
};

// Register the reducer with ReducerRegistry
ReducerRegistry.register(`features/${SIGN_LANGUAGE_FEATURE}`, signLanguageReducer);

// Optionally persist the state (uncomment if persistence is desired)
// PersistenceRegistry.register(`features/${SIGN_LANGUAGE_FEATURE}`);

// Type for a single frame of 126 keypoints (21 landmarks × 3 coordinates per hand)
type KeypointFrame = number[] & { length: 126 };

// Type for a sequence of keypoint frames
type KeypointSequence = KeypointFrame[];

// Constants for keypoint processing
const LANDMARK_COUNT = 21;
const COORDS_PER_LANDMARK = 3;
const KEYPOINTS_PER_FRAME = LANDMARK_COUNT * COORDS_PER_LANDMARK * 2; // 126
const WRIST_INDEX = 0;
const MIDDLE_MCP_INDEX = 9;

/**
 * Normalizes a sequence of keypoint frames for hand pose analysis.
 * Each frame contains 126 numbers (21 landmarks × 3 coordinates for left hand,
 * followed by right hand). For each hand:
 * - Centers coordinates by subtracting the wrist (landmark 0).
 * - Scales by the Euclidean distance of the middle finger MCP (landmark 9) from origin.
 * - Returns zeros if the hand is undetected (all zeros) or scaling fails.
 *
 * @param sequence Array of frames, each with 126 numbers.
 * @returns Normalized sequence with same structure.
 * @throws Error if a frame has incorrect length.
 */
const normalizeKeypoints = (sequence: KeypointSequence): KeypointSequence => {
  const normalizedSequence: KeypointSequence = [];

  for (const frame of sequence) {
    if (frame.length !== KEYPOINTS_PER_FRAME) {
      throw new Error(`Invalid frame length: expected ${KEYPOINTS_PER_FRAME}, got ${frame.length}`);
    }

    const normalizedFrame = new Array(KEYPOINTS_PER_FRAME).fill(0) as KeypointFrame;

    // Process left hand (indices 0–62)
    let hasLeftHand = false;
    for (let i = 0; i < LANDMARK_COUNT * COORDS_PER_LANDMARK; i++) {
      if (frame[i] !== 0) {
        hasLeftHand = true;
        break;
      }
    }

    if (hasLeftHand) {
      // Center using wrist (index 0)
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

      // Calculate hand size using middle finger MCP (index 9)
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
    let hasRightHand = false;
    for (let i = LANDMARK_COUNT * COORDS_PER_LANDMARK; i < KEYPOINTS_PER_FRAME; i++) {
      if (frame[i] !== 0) {
        hasRightHand = true;
        break;
      }
    }

    if (hasRightHand) {
      // Center using wrist (index 21)
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

      // Calculate hand size using middle finger MCP (index 30)
      const mcpBase = (MIDDLE_MCP_INDEX * COORDS_PER_LANDMARK) + wristBase;
      const handSize = Math.sqrt(
        centeredRight[MIDDLE_MCP_INDEX * COORDS_PER_LANDMARK] ** 2 +
        centeredRight[MIDDLE_MCP_INDEX * COORDS_PER_LANDMARK + 1] ** 2 +
        centeredRight[MIDDLE_MCP_INDEX * COORDS_PER_LANDMARK + 2] ** 2
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
  const holisticRef = useRef<Holistic | null>(null);
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
  }, [t]);

  // Initialize MediaPipe Holistic and TensorFlow model
  useEffect(() => {
    let isMounted = true;

    const initialize = async () => {
      try {
        // Initialize MediaPipe Holistic
        const holistic = new Holistic({
          locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/holistic@0.5.1675471629/${file}`,
        });
        holistic.setOptions({
          modelComplexity: 0,
          selfieMode: true,
          smoothLandmarks: true,
          enableSegmentation: false,
          enableFaceLandmarks: false,
          enablePoseLandmarks: false,
          minDetectionConfidence: 0.5,
          minTrackingConfidence: 0.5,
        });
        await holistic.initialize();
        if (!isMounted) return;
        holisticRef.current = holistic;
        console.log('MediaPipe Holistic initialized');

        // Set TensorFlow backend
        const backend = await tf.setBackend('webgl') ? 'webgl' : 'cpu';
        console.log('TensorFlow.js backend:', backend);

        // Load model
        const modelPath = `static/sign_language_model_tfjs/model.json`;
        const loadedModel = await tf.loadLayersModel(modelPath, {
          onProgress: (fraction) => console.log('Model loading progress:', fraction),
        });

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
        console.log('Model and labels loaded');
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
      if (holisticRef.current) {
        holisticRef.current.close();
        holisticRef.current = null;
      }
      if (model) {
        model.dispose();
        setModel(null);
      }
    };
  }, [t]);

  // Extract keypoints
  const extractKeypoints = (results: any): number[] => {
    const leftHand = results.leftHandLandmarks
      ? results.leftHandLandmarks.map((landmark: any) => [landmark.x, landmark.y, landmark.z]).flat()
      : new Array(21 * 3).fill(0);
    const rightHand = results.rightHandLandmarks
      ? results.rightHandLandmarks.map((landmark: any) => [landmark.x, landmark.y, landmark.z]).flat()
      : new Array(21 * 3).fill(0);
    return [...leftHand, ...rightHand];
  };

  // Preprocess frame
  const preprocessFrame = async (imageData: ImageData): Promise<tf.Tensor> => {
    if (!holisticRef.current) {
      console.error('Holistic not initialized');
      return tf.zeros([1, 30, 126]);
    }

    const canvas = document.createElement('canvas');
    canvas.width = 320;
    canvas.height = 240;
    const ctx = canvas.getContext('2d');
    if (!ctx) throw new Error('No canvas context');
    ctx.putImageData(imageData, 0, 0);

    console.time('Holistic');
    const results = await new Promise<any>((resolve) => {
      holisticRef.current!.onResults(resolve);
      holisticRef.current!.send({ image: canvas });
    });
    console.timeEnd('Holistic');

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
    if (!model) {
      console.error('Model not loaded');
      return 'Model not loaded';
    }
    if (!labels.length) {
      console.error('Labels not loaded');
      return 'Labels not loaded';
    }
    try {
      console.time('Inference');
      const prediction = model.predict(tensor) as tf.Tensor;
      const [probs, labelIndexTensor] = await Promise.all([
        prediction.data(),
        prediction.argMax(-1).data(),
      ]);
      const labelIndex = labelIndexTensor[0];
      const confidence = probs[labelIndex];
      prediction.dispose();
      console.timeEnd('Inference');
      return confidence > 0.7 ? labels[labelIndex] : 'Unknown';
    } catch (error) {
      console.error('Prediction error:', error);
      return 'Prediction failed';
    }
  };

  // Update subtitles
  const updateSubtitles = (text: string) => {
    console.log('Dispatching action:', { type: 'UPDATE_SIGN_LANGUAGE_SUBTITLES', text });
    if (APP.store) {
      APP.store.dispatch({
        type: 'UPDATE_SIGN_LANGUAGE_SUBTITLES',
        text: text,
      });
      console.log('Store state after dispatch:', APP.store.getState());
    } else {
      console.warn('APP.store not initialized, cannot dispatch UPDATE_SIGN_LANGUAGE_SUBTITLES');
    }
  };

  // Get the local video stream
  const getLocalVideoStream = (): any | null => {
    console.log('getLocalVideoStream called');
    const conference = APP.conference._room;
    if (!conference) {
      console.log('No conference room available');
      return null;
    }

    const localId = conference.myUserId();
    console.log('Local ID:', localId);

    if (!localId) {
      console.log('No local ID available');
      return null;
    }

    const localTracks = conference.getLocalTracks();
    console.log('Local tracks:', localTracks);

    if (!localTracks || localTracks.length === 0) {
      console.log('No local tracks available');
      return null;
    }

    const videoTrack = localTracks.find(track => track.getType() === 'video');
    if (!videoTrack) {
      console.log('No video track found');
      return null;
    }

    console.log('Video track retrieved:', videoTrack);
    return videoTrack;
  };

  // Extract a single frame using ImageCapture
  const extractFrame = async (videoTrack: any): Promise<ImageData> => {
    console.log('Starting frame extraction with ImageCapture');
    if (!videoTrack) {
      throw new Error('No video track provided');
    }
    if (!videoTrack.isVideoTrack()) {
      throw new Error('Track is not a video track');
    }
    if (videoTrack.isMuted()) {
      throw new Error('Video track is muted');
    }
    if (videoTrack.videoType !== 'camera') {
      throw new Error('Track is not a camera track');
    }

    const track = videoTrack.getTrack();
    console.log('Track:', track, 'readyState:', track?.readyState, 'muted:', track?.muted);
    if (!track || track.readyState !== 'live') {
      throw new Error('No valid MediaStreamTrack');
    }

    try {
      const imageCapture = new ImageCapture(track);
      console.log('ImageCapture created');
      const bitmap = await imageCapture.grabFrame();
      console.log('Bitmap captured:', bitmap);

      const canvas = document.createElement('canvas');
      canvas.width = 320;
      canvas.height = 240;
      const ctx = canvas.getContext('2d');
      if (!ctx) {
        throw new Error('No canvas context');
      }
      ctx.drawImage(bitmap, 0, 0, 320, 240);
      const imageData = ctx.getImageData(0, 0, 320, 240);

      console.log('Frame extracted with ImageCapture:', imageData);
      return imageData;
    } catch (error) {
      console.error('ImageCapture error:', error);
      throw error;
    }
  };

  // Process frames continuously using ImageCapture
  const processVideoFrames = (videoTrack: any, callback: (data: ImageData) => void): () => void => {
    let animationFrameId: number | null = null;
    let shouldContinue = true;

    const processFrame = async (timestamp: number) => {
      if (!shouldContinue) {
        console.log('Stopping frame processing: translation disabled');
        return;
      }
      if (videoTrack.isMuted()) {
        console.log('Stopping frame processing: video track muted');
        return;
      }
      if (timestamp - lastFrameTimeRef.current < 66) {
        animationFrameId = requestAnimationFrame(processFrame);
        return;
      }
      lastFrameTimeRef.current = timestamp;

      try {
        const startTime = performance.now();
        const imageData = await extractFrame(videoTrack);
        callback(imageData);

        const tensor = await preprocessFrame(imageData);
        console.log('Tensor shape:', tensor.shape);

        if (sequenceRef.current.length < 30) {
          console.log('Waiting for 30 frames:', sequenceRef.current.length);
          tensor.dispose();
          animationFrameId = requestAnimationFrame(processFrame);
          return;
        }

        const prediction = await predictSign(tensor);
        console.log('Predicted sign:', prediction);

        predictionsRef.current.push(prediction);
        if (predictionsRef.current.length > 10) {
          predictionsRef.current.shift();
        }
        const recentPredictions = predictionsRef.current;
        const isStable = recentPredictions.length === 10 &&
          recentPredictions.every(p => p === prediction);

        if (isStable && prediction !== 'Unknown') {
          updateSubtitles(prediction);
          console.log('Subtitles updated:', prediction);
        }

        tensor.dispose();
        console.log('Frame processing time:', performance.now() - startTime);
      } catch (error) {
        console.error('Error processing frame:', error);
        APP.store.dispatch({ type: 'SET_SIGN_LANGUAGE_ERROR', error: 'errors.initializationFailed' });
      }

      animationFrameId = requestAnimationFrame(processFrame);
    };

    console.log('Continuous processing started');
    requestAnimationFrame(processFrame);

    return () => {
      shouldContinue = false;
      if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
      }
      sequenceRef.current = [];
      predictionsRef.current = [];
      console.log('Frame processing stopped');
    };
  };

  // Wait for conference
  const waitForConference = () => {
    return new Promise<void>((resolve) => {
      if (APP.conference && APP.conference.isJoined()) {
        console.log('Conference already joined');
        resolve();
      } else {
        console.log('Waiting for conference to join');
        APP.conference.on(JitsiMeetJS.events.conference.CONFERENCE_JOINED, () => {
          console.log('Conference joined');
          resolve();
        });
      }
    });
  };

  const handleClick = async () => {
    if (!isDeviceSupported) {
      APP.store.dispatch({ type: 'SET_SIGN_LANGUAGE_ERROR', error: 'errors.deviceUnsupported' });
      return;
    }
    console.log('SignLanguageButton clicked');
    setIsTranslationEnabled(prev => {
      const newValue = !prev;
      console.log('Translation enabled:', newValue);

      if (newValue) {
        console.log('Starting translation process');
        waitForConference()
          .then(() => {
            const attemptGetStream = async (attempts = 3, delay = 500): Promise<void> => {
              console.log('Attempting to get video stream, attempt:', attempts);
              let videoTrack: any | null = null;
              try {
                videoTrack = getLocalVideoStream();
              } catch (error) {
                console.error('Error in getLocalVideoStream:', error);
                APP.store.dispatch({ type: 'SET_SIGN_LANGUAGE_ERROR', error: 'errors.initializationFailed' });
                return;
              }

              if (!videoTrack) {
                console.log('No video track found');
              } else {
                console.log('Is video track:', videoTrack.isVideoTrack());
                console.log('Video type:', videoTrack.videoType);
                console.log('Track readyState:', videoTrack.getTrack()?.readyState);
                console.log('Track muted:', videoTrack.isMuted());
              }

              const condition = videoTrack && !videoTrack.isMuted() && videoTrack.isVideoTrack();
              console.log('Condition:', condition);

              if (condition) {
                try {
                  console.log('Calling extractFrame');
                  const frame = await extractFrame(videoTrack);
                  console.log('Initial frame extracted:', frame);

                  animationFrameRef.current = processVideoFrames(videoTrack, (imageData) => {
                    console.log('Processing frame:', imageData);
                  });
                } catch (error) {
                  console.error('Error extracting initial frame:', error);
                  APP.store.dispatch({ type: 'SET_SIGN_LANGUAGE_ERROR', error: 'errors.initializationFailed' });
                }
              } else if (attempts > 0) {
                console.log(`Retrying to get video stream (${attempts} attempts left)`);
                await new Promise(resolve => setTimeout(resolve, delay));
                return attemptGetStream(attempts - 1, delay);
              } else {
                console.error('Failed to get local video stream after retries');
                APP.store.dispatch({ type: 'SET_SIGN_LANGUAGE_ERROR', error: 'errors.initializationFailed' });
              }
            };

            attemptGetStream().catch(error => {
              console.error('Error in attemptGetStream:', error);
              APP.store.dispatch({ type: 'SET_SIGN_LANGUAGE_ERROR', error: 'errors.initializationFailed' });
            });
          })
          .catch(error => {
            console.error('Error waiting for conference:', error);
            APP.store.dispatch({ type: 'SET_SIGN_LANGUAGE_ERROR', error: 'errors.initializationFailed' });
          });
      } else {
        console.log('Stopping translation process');
        if (animationFrameRef.current) {
          animationFrameRef.current();
          animationFrameRef.current = null;
        }
      }

      return newValue;
    });
  };

  console.log('SignLanguageButton rendered');

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
  (state: any) => ({}),
  (dispatch: any) => ({})
)(SignLanguageButton));

// SignLanguageOverlay component
interface OverlayProps extends WithTranslation {
  subtitles: string;
  error: string | null;
}

const SignLanguageOverlay: React.FC<OverlayProps> = ({ subtitles, error, t }) => {
  const [displayText, setDisplayText] = useState('');

  useEffect(() => {
    console.log('Subtitles prop updated:', subtitles, 'Error:', error);
    if (error) {
      setDisplayText(t(error));
      return;
    }
    if (subtitles.trim()) {
      setDisplayText(subtitles);
      const timeout = setTimeout(() => setDisplayText(''), 3000);
      return () => clearTimeout(timeout);
    }
  }, [subtitles, error, t]);

  useEffect(() => {
    if (!APP.store) {
      console.warn('APP.store not initialized, cannot subscribe to store updates');
      return;
    }
    const unsubscribe = APP.store.subscribe(() => {
      console.log('Store updated:', APP.store.getState());
    });
    return () => unsubscribe();
  }, []);

  if (!displayText) {
    console.log('Overlay not rendering: displayText is empty');
    return null;
  }

  console.log('Overlay rendering with displayText:', displayText);

  return (
    <>
      <style>
        {`
          @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
          }
        `}
      </style>
      <div
        style={{
          position: 'fixed', // Changed to fixed for viewport positioning
          bottom: '3rem',
          left: '50%',
          transform: 'translateX(-50%)',
          zIndex: 99999, // Increased
          opacity: 0,
          animation: 'fadeIn 0.3s ease-in forwards',
          border: '2px solid red', // Debug
          pointerEvents: 'none', // Prevent click interference
          backgroundColor: 'rgba(255, 0, 0, 0.1)', // Debug background
        }}
        aria-live="polite"
      >
        <div
          style={{
            backgroundColor: 'rgba(0, 0, 0, 0.7)',
            color: 'white',
            fontSize: '1.5rem',
            fontWeight: 'bold',
            padding: '0.5rem 1rem',
            borderRadius: '0.5rem',
            boxShadow: '0 2px 4px rgba(0, 0, 0, 0.2)',
          }}
        >
          {t('signLanguage.prediction')}: {displayText}
        </div>
      </div>
    </>
  );
};

const ConnectedSignLanguageOverlay = withTranslation()(connect(
  (state: any) => {
    console.log('Overlay state:', state);
    return {
      subtitles: state['features/sign-language']?.text || '',
      error: state['features/sign-language']?.error || null,
    };
  }
)(SignLanguageOverlay));

const SignLanguageOverlayApp: React.FC = () => {
  console.log('SignLanguageOverlayApp rendered');
  return <ConnectedSignLanguageOverlay />;
};

// Update SignLanguageApp to remove Provider
const SignLanguageApp: React.FC = () => {
  useEffect(() => {
    if (!APP.store) {
      console.warn('APP.store not initialized, skipping SignLanguageApp initialization');
      return;
    }
    console.log('APP.store:', APP.store);
    console.log('Store keys:', Object.keys(APP.store.getState().features));
    if (process.env.NODE_ENV === 'development') {
      APP.store.dispatch({ type: 'UPDATE_SIGN_LANGUAGE_SUBTITLES', text: 'TEST' });
    }
  }, []);
  return <ConnectedSignLanguageButton />;
};


export { SignLanguageApp, SignLanguageOverlayApp, ConnectedSignLanguageButton };
export default SignLanguageApp;