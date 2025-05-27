import ReducerRegistry from '../redux/ReducerRegistry';
import { TOGGLE_SIGN_LANGUAGE_TRANSCRIPTION, UPDATE_SIGN_LANGUAGE_TEXT } from './actionTypes';

const initialState = {
    isTranscribing: false, // Whether the current user is transcribing sign language
    transcriptions: {} // Map of userId to transcribed text (e.g., { userId: "Hello" })
};

ReducerRegistry.register('features/base/sign-language', (state = initialState, action) => {
    switch (action.type) {
        case TOGGLE_SIGN_LANGUAGE_TRANSCRIPTION:
            return {
                ...state,
                isTranscribing: !state.isTranscribing
            };
        case UPDATE_SIGN_LANGUAGE_TEXT:
            return {
                ...state,
                transcriptions: {
                    ...state.transcriptions,
                    [action.userId]: action.text
                }
            };
        default:
            return state;
    }
});