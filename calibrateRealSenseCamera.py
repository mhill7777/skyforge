import pyrealsense2 as rs

# 1. Get current calibration
cal_before = rs.rs2_get_calibration_table()

# 2. Run on-chip calibration
cal_after = rs.rs2_run_on_chip_calibration()

# 3. Optionally test new calibration
rs.rs2_set_calibration_table(cal_after)

# 4. Write new calibration to firmware
rs.rs2_write_calibration()
