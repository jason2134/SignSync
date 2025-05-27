import { TOGGLE_SIGN_LANGUAGE_TRANSCRIPTION, UPDATE_SIGN_LANGUAGE_TEXT } from './actionTypes';

export function toggleSignLanguageTranscription() {
    return {
        type: TOGGLE_SIGN_LANGUAGE_TRANSCRIPTION
    };
}

export function updateSignLanguageText(userId, text) {
    return {
        type: UPDATE_SIGN_LANGUAGE_TEXT,
        userId,
        text
    };
}