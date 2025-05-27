import React, { Component } from 'react';
import { FaBeer, FaHome } from 'react-icons/fa';
import { translate } from '../../../base/i18n/functions';
//import { translate } from '../../../base/i18n';
//import { IconSignLanguage } from '../../../base/icons'; // Use an existing icon or add a new one
import AbstractButton from '../../../base/toolbox/components/AbstractButton';

import { WithTranslation, withTranslation } from 'react-i18next';

function getPinnedVideoStream() {
    const conference = APP.conference._room; // Access the current conference
    const pinnedParticipant = conference.getPinnedParticipant();
    if (pinnedParticipant) {
        const tracks = conference.getParticipantTracks(pinnedParticipant.id);
        return tracks.find(track => track.type === 'video'); // Get video track
    }
    return null;
}

