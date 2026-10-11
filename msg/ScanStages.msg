std_msgs/Header header
float64 deskew_ms
float64 voxel_ms
float64 align_ms
float64 cost_ms
float64 refine_ms
float64 map_ms
bool timed_out
bool skipped
bool applied
# Scans dropped before a pose update: empty or uncovered scans, plus rate-budget timeouts.
uint32 skipped_total
