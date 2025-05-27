import MiddlewareRegistry from '../redux/MiddlewareRegistry';
import { TOGGLE_SIGN_LANGUAGE_TRANSCRIPTION, updateSignLanguageText } from './actions';
import { performSignLanguageRecognition } from './recognition';

let frameCaptureInterval = null;
let videoElement = null;
let canvas = null;

const middleware = store => next => action => {
    const result = next(action);

    if (action.type === TOGGLE_SIGN_LANGUAGE_TRANSCRIPTION) {
        const state = store.getState();
        const isTranscribing = state['features/base/sign-language'].isTranscribing;

        if (isTranscribing) {
            const conference = APP.conference._room;
            const localTracks = conference.getLocalTracks();
            const videoTrack = localTracks.find(track => track.getType() === 'video');

            if (videoTrack) {
                const stream = videoTrack.getTrack().getStream();
                videoElement = document.createElement('video');
                videoElement.srcObject = stream;
                videoElement.play();

                canvas = document.createElement('canvas');
                canvas.width = videoElement.videoWidth;
                canvas.height = videoElement.videoHeight;

                frameCaptureInterval = setInterval(async () => {
                    const ctx = canvas.getContext('2d');
                    ctx.drawImage(videoElement, 0, 0);
                    const frame = canvas.toDataURL('image/jpeg');
                    

                    const recognizedText = await performSignLanguageRecognition(frame);
                    if (recognizedText) {
                        conference.sendCommand('signLanguage', {
                            value: recognizedText,
                            attributes: { sender: conference.myUserId() }
                        });
                        store.dispatch(updateSignLanguageText(conference.myUserId(), recognizedText));
                    }
                }, 500);
            }
        } else {
            if (frameCaptureInterval) {
                clearInterval(frameCaptureInterval);
                frameCaptureInterval = null;
            }
            if (videoElement) {
                videoElement.srcObject = null;
                videoElement = null;
            }
            if (canvas) {
                canvas = null;
            }
        }
    }

    return result;
};

// Listen for sign language transcriptions from other participants
const setupSignLanguageListener = () => {
    APP.conference._room.on('command:signLanguage', ({ value, attributes }) => {
        const { sender } = attributes;
        APP.store.dispatch(updateSignLanguageText(sender, value));
    });
};

APP.on('conferenceJoined', setupSignLanguageListener);

// Register the middleware
MiddlewareRegistry.register(middleware);

export default middleware;