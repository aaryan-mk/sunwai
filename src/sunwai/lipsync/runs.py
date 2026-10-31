"""Pure run-splitting helper for per-frame boolean masks (no model dependencies)."""


def split_into_runs(face_mask, fps, min_run_seconds=1.0, bridge_frames=3):
    """
    Split boolean mask into runs of faces.
    Returns list of (start_frame, end_frame) tuples (inclusive of start, exclusive of end?
    Let's use start_idx, end_idx where end_idx is exclusive like slicing, or start_sec, end_sec?
    Let's return frame indices [start, end) for easy slicing and conversion to time.
    """
    if not face_mask:
        return []

    n_frames = len(face_mask)
    if n_frames == 0:
        return []

    # 1. Find contiguous blocks of True
    blocks = []
    in_run = False
    start = 0

    for i, has_face in enumerate(face_mask):
        if has_face and not in_run:
            in_run = True
            start = i
        elif not has_face and in_run:
            in_run = False
            blocks.append([start, i])
    if in_run:
        blocks.append([start, n_frames])

    # 2. Bridge small gaps
    if not blocks:
        return []

    bridged_blocks = [blocks[0]]
    for i in range(1, len(blocks)):
        prev_end = bridged_blocks[-1][1]
        curr_start = blocks[i][0]

        gap = curr_start - prev_end

        if gap <= bridge_frames:
            # Bridge them
            bridged_blocks[-1][1] = blocks[i][1]
        else:
            bridged_blocks.append(blocks[i])

    # 3. Filter by min duration and convert to times
    runs = []
    min_frames = int(min_run_seconds * fps)

    for start, end in bridged_blocks:
        duration_frames = end - start
        if duration_frames >= min_frames:
            runs.append([start, end])
        # else: discard small runs

    return runs
