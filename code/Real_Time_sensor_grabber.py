import requests
import time

# One combined URL for ALL sensors listed using inspect element for JSON
URL = (
    "http://172.20.10.1/get?"
    "accX=full&acc_time=full&accY=full&accZ=full&"
    "gyroX=full&gyro_time=full&gyroY=full&gyroZ=full&"
    "graX=full&graT=full&graY=full&graZ=full&"
    "lin_accX=full&lin_acc_time=full&lin_accY=full&lin_accZ=full"
)

def last_value(buffers, name):
    """Safely get last value from a PhyPhox buffer."""
    ch = buffers.get(name)
    if not ch:
        return None
    buf = ch.get("buffer", [])
    return buf[-1] if buf else None

while True:
    try:
        r = requests.get(URL, timeout=2)
        data = r.json()
        buffers = data["buffer"]

        # --- accelerometer ---
        ax = last_value(buffers, "accX")
        ay = last_value(buffers, "accY")
        az = last_value(buffers, "accZ")

        # --- gyroscope ---
        gx = last_value(buffers, "gyroX")
        gy = last_value(buffers, "gyroY")
        gz = last_value(buffers, "gyroZ")

        # --- gravity ---
        gvx = last_value(buffers, "graX")
        gvy = last_value(buffers, "graY")
        gvz = last_value(buffers, "graZ")

        # --- linear acceleration ---
        lax = last_value(buffers, "lin_accX")
        lay = last_value(buffers, "lin_accY")
        laz = last_value(buffers, "lin_accZ")

        # Only print when we have data for everything
        if None not in (ax, ay, az, gx, gy, gz, gvx, gvy, gvz, lax, lay, laz):
            print(
                f"ACC  : X={ax:7.3f} Y={ay:7.3f} Z={az:7.3f}\n"
                f"GYRO : X={gx:7.3f} Y={gy:7.3f} Z={gz:7.3f}\n"
                f"GRAV : X={gvx:7.3f} Y={gvy:7.3f} Z={gvz:7.3f}\n"
                f"LIN A: X={lax:7.3f} Y={lay:7.3f} Z={laz:7.3f}\n"
                "--------------------------------------------------"
            )
        else:
            print("Waiting for all sensor buffers to fill...")

    except Exception as e:
        print("Error:", e)

    time.sleep(0.1)   # 10 updates per second
