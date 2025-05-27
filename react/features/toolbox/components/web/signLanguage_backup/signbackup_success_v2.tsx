import React, { useState, useEffect, useRef } from 'react';
import { FaBeer } from 'react-icons/fa';
import { translate } from '../../../../base/i18n/functions';
import { withTranslation, WithTranslation } from 'react-i18next';
import { connect } from 'react-redux';
import * as tf from '@tensorflow/tfjs';
import { Holistic } from '@mediapipe/holistic';
import '@tensorflow/tfjs-backend-webgl';
import JitsiMeetJS from 'lib-jitsi-meet';

interface IProps extends WithTranslation {
  'aria-label'?: string;
  className?: string;
}

const SignLanguageButton: React.FC<IProps> = ({ t, 'aria-label': ariaLabel, className }) => {
  const [isTranslationEnabled, setIsTranslationEnabled] = useState(false);
  const animationFrameRef = useRef<(() => void) | null>(null);
  const [model, setModel] = useState<tf.LayersModel | null>(null);
  const [labels, setLabels] = useState<string[]>([]);
  const holisticRef = useRef<Holistic | null>(null);
  const sequenceRef = useRef<number[][]>([]);
  const predictionsRef = useRef<string[]>([]);
  const lastFrameTimeRef = useRef<number>(0);

  // Initialize MediaPipe Holistic and TensorFlow model
  useEffect(() => {
    const initializeHolistic = async () => {
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
      holisticRef.current = holistic;
      console.log('MediaPipe Holistic initialized');
    };

    const loadModel = async () => {
      try {
        let backendSet = await tf.setBackend('webgl');
        if (!backendSet) {
          console.warn('WebGL backend not available, falling back to CPU');
          await tf.setBackend('cpu');
        }
        console.log('TensorFlow.js backend:', tf.getBackend());
        console.log('TensorFlow.js version:', tf.version.tfjs);

        const modelPath = `static/sign_language_model_tfjs/model.json`;
        const loadedModel = await tf.loadLayersModel(modelPath, {
          onProgress: (fraction) => console.log('Model loading progress:', fraction),
        });

        // Warm up model
        const dummyInput = tf.zeros([1, 30, 126]);
        loadedModel.predict(dummyInput).dispose();
        dummyInput.dispose();

        setModel(loadedModel);
        console.log('Model loaded successfully:', loadedModel);

        const signLanguageLabels = [
          'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
          'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
        ];
        setLabels(signLanguageLabels);
      } catch (error) {
        console.error('Failed to load TensorFlow model:', error);
      }
    };

    initializeHolistic();
    loadModel();

    return () => {
      if (animationFrameRef.current) {
        animationFrameRef.current();
        animationFrameRef.current = null;
      }
      if (holisticRef.current) {
        holisticRef.current.close();
      }
    };
  }, []);

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

  // Normalize keypoints (original JavaScript version)
  const normalizeKeypoints = (sequence: number[][]): number[][] => {
    const normalizedSequence: number[][] = [];

    for (const frame of sequence) {
      const frameArray = [];
      for (let i = 0; i < frame.length; i += 3) {
        frameArray.push([frame[i], frame[i + 1], frame[i + 2]]);
      }

      let leftHand = frameArray.slice(0, 21);
      if (leftHand.some(coord => coord.some(val => val !== 0))) {
        const leftHandCenter = leftHand[0];
        leftHand = leftHand.map(coord => [
          coord[0] - leftHandCenter[0],
          coord[1] - leftHandCenter[1],
          coord[2] - leftHandCenter[2],
        ]);
        const handSize = Math.sqrt(
          leftHand[9].reduce((sum, val) => sum + (val ** 2), 0),
        );
        if (handSize > 0) {
          leftHand = leftHand.map(coord => [
            coord[0] / handSize,
            coord[1] / handSize,
            coord[2] / handSize,
          ]);
        } else {
          leftHand = new Array(21).fill(0).map(() => [0, 0, 0]);
        }
      } else {
        leftHand = new Array(21).fill(0).map(() => [0, 0, 0]);
      }

      let rightHand = frameArray.slice(21, 42);
      if (rightHand.some(coord => coord.some(val => val !== 0))) {
        const rightHandCenter = rightHand[0];
        rightHand = rightHand.map(coord => [
          coord[0] - rightHandCenter[0],
          coord[1] - rightHandCenter[1],
          coord[2] - rightHandCenter[2],
        ]);
        const handSize = Math.sqrt(
          rightHand[9].reduce((sum, val) => sum + (val ** 2), 0),
        );
        if (handSize > 0) {
          rightHand = rightHand.map(coord => [
            coord[0] / handSize,
            coord[1] / handSize,
            coord[2] / handSize,
          ]);
        } else {
          rightHand = new Array(21).fill(0).map(() => [0, 0, 0]);
        }
      } else {
        rightHand = new Array(21).fill(0).map(() => [0, 0, 0]);
      }

      const frameNormalized = [...leftHand, ...rightHand].flat();
      normalizedSequence.push(frameNormalized);
    }

    return normalizedSequence;
  };

  // Preprocess frame (fixed drawImage error)
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
      const normalizedSequence = normalizeKeypoints(paddedSequence);
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
    APP.store.dispatch({
      type: 'UPDATE_SUBTITLES',
      text: text,
    });
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
                }
              } else if (attempts > 0) {
                console.log(`Retrying to get video stream (${attempts} attempts left)`);
                await new Promise(resolve => setTimeout(resolve, delay));
                return attemptGetStream(attempts - 1, delay);
              } else {
                console.error('Failed to get local video stream after retries');
                alert('Please enable your webcam to use sign language translation.');
              }
            };

            attemptGetStream().catch(error => {
              console.error('Error in attemptGetStream:', error);
            });
          })
          .catch(error => {
            console.error('Error waiting for conference:', error);
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
    >
      {t('Sign Language')}
    </button>
  );
};

const mapStateToProps = (state: any) => ({});
const mapDispatchToProps = (dispatch: any) => ({});

export default translate(connect(mapStateToProps, mapDispatchToProps)(SignLanguageButton));