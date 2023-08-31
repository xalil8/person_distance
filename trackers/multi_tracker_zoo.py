from trackers.parser import get_config

def create_tracker(tracker_type, tracker_config, reid_weights, device, half):
    
    cfg = get_config()
    cfg.merge_from_file(tracker_config)

    
    if tracker_type == 'ocsort':
        from trackers.ocsort.ocsort import OCSort
        ocsort = OCSort(
            det_thresh=cfg.ocsort.det_thresh,
            max_age=cfg.ocsort.max_age,
            min_hits=cfg.ocsort.min_hits,
            iou_threshold=cfg.ocsort.iou_thresh,
            delta_t=cfg.ocsort.delta_t,
            asso_func=cfg.ocsort.asso_func,
            inertia=cfg.ocsort.inertia,
            use_byte=cfg.ocsort.use_byte,
        )
        return ocsort
    
    elif tracker_type == 'bytetrack':
        from trackers.bytetrack.byte_tracker import BYTETracker
        bytetracker = BYTETracker(
            track_thresh=cfg.bytetrack.track_thresh,
            match_thresh=cfg.bytetrack.match_thresh,
            track_buffer=cfg.bytetrack.track_buffer,
            frame_rate=cfg.bytetrack.frame_rate
        )
        return bytetracker

        return botsort

    else:
        print('No such tracker')
        exit()