from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import time
from typing import Dict, Any

app = FastAPI(title="Sensor Logger HTTP Push Receiver")

@app.post("/data")
async def receive_sensor_data(request: Request):
    try:
        payload = await request.json()
        # Sensor Logger typical payload structure:
        # {
        #   "messageId": int,
        #   "sessionId": str,
        #   "deviceId": str,
        #   "payload": [
        #     {"name": "accelerometer", "values": [x,y,z], "timestamp": float},
        #     {"name": "gyroscope",     "values": [x,y,z], "timestamp": float},
        #     {"name": "magnetometer",  "values": [x,y,z], "timestamp": float},
        #     ...
        #   ]
        # }

        # Quick extraction (customize based on your enabled sensors)
        accel = next((p for p in payload.get("payload", []) if p["name"] == "accelerometer"), None)
        gyro  = next((p for p in payload.get("payload", []) if p["name"] == "gyroscope"), None)

        timestamp = time.strftime("%H:%M:%S.%f")[:-3]  # local time with ms

        if accel and gyro:
            ax, ay, az = accel["values"]
            gx, gy, gz = gyro["values"]
            print(f"[{timestamp}] "
                  f"Accel: {ax:+6.3f} {ay:+6.3f} {az:+6.3f} g | "
                  f"Gyro:  {gx:+7.2f} {gy:+7.2f} {gz:+7.2f} °/s")

        # Return 200 OK so Sensor Logger keeps sending
        return JSONResponse(status_code=200, content={"status": "received"})

    except Exception as e:
        print("Error processing payload:", e)
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/")
def root():
    return {"message": "Sensor Logger receiver is running. Send POST to /data"}

if __name__ == "__main__":
    print("Starting Sensor Logger HTTP receiver...")
    print("Make sure iPhone push URL is: http://192.168.1.99:8000/data")
    print("Press Ctrl+C to stop")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")