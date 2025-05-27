import { IStore } from "./react/features/app/types";
import { IConfig } from "./react/features/base/config/configType";

export {};

// Define an interface for the conference object
// Placeholder for JitsiTrack (refine if you have lib-jitsi-meet types)
interface JitsiTrack {
    getType: () => string;
    getTrack: () => MediaStreamTrack;
    // Add other properties/methods as needed
}

// Placeholder for JitsiParticipant
interface JitsiParticipant {
    getId: () => string;
    getTracks: () => JitsiTrack[];
    // Add other properties/methods as needed
}

interface IConference {
    getPinnedVideoStream: () => JitsiTrack | null;
    _room: {
        getParticipantById: (id: string) => JitsiParticipant | undefined;
        // Add other JitsiConference methods as needed
    };
    init: (options: { roomName: string; shouldDispatchConnect?: boolean }) => Promise<void>;
    _createRoom: (localTracks: any[]) => void;
}

declare global {
    const APP: {
        store: IStore;
        UI: any;
        API: any;
        conference: any;
        debugLogs: any;
    };
    const interfaceConfig: any;

    interface Window {
        config: IConfig;
        JITSI_MEET_LITE_SDK?: boolean;
        interfaceConfig?: any;
        JitsiMeetJS?: any;
        PressureObserver?: any;
        PressureRecord?: any;
        ReactNativeWebView?: any;
        // selenium tests handler
        _sharedVideoPlayer: any;
        alwaysOnTop: { api: any };
    }

    interface Document {
        mozCancelFullScreen?: Function;
        webkitExitFullscreen?: Function;
    }

    const config: IConfig;

    const JitsiMeetJS: any;

    interface HTMLMediaElement {
        setSinkId: (id: string) => Promise<undefined>;
        stop: () => void;
    }
}
