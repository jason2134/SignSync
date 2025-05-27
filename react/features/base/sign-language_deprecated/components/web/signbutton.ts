import React from 'react';
import { connect } from 'react-redux';
import { toggleSignLanguageTranscription } from '../../actions';

const SignButton = ({ isTranscribing, toggleSignLanguageTranscription }) => (
    <button
        onClick={toggleSignLanguageTranscription}
        className={`toolbox-button ${isTranscribing ? 'active' : ''}`}
        title={isTranscribing ? 'Stop Sign Language Transcription' : 'Start Sign Language Transcription'}
    >
        <span className="icon-sign-language">✋</span> {}
    </button>
);

const mapStateToProps = state => ({
    isTranscribing: state['features/base/sign-language'].isTranscribing
});

export default connect(mapStateToProps, { toggleSignLanguageTranscription })(SignButton);