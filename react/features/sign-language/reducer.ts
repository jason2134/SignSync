import { AnyAction } from 'redux';
import ReducerRegistry from '../base/redux/ReducerRegistry';

export interface ISignLanguageState {
  text: string;
}

const initialState: ISignLanguageState = { text: '' };

const SIGN_LANGUAGE_FEATURE = 'sign-language';

const reducer = (state = initialState, action: AnyAction): ISignLanguageState => {
  console.log('signLanguageReducer called with action:', action);
  switch (action.type) {
    case 'UPDATE_SIGN_LANGUAGE_SUBTITLES':
      console.log('Updating sign language state to:', { text: action.text || '' });
      return { ...state, text: action.text || '' };
    default:
      return state;
  }
};

// Register the reducer with ReducerRegistry
ReducerRegistry.register(`features/${SIGN_LANGUAGE_FEATURE}`, reducer);

export default reducer;