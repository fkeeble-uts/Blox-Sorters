# Start the Dobot Magician via PuTTY (Raspberry Pi) + Run Python Control Script

This guide shows how to SSH into your Raspberry Pi with **PuTTY**

---

## Network & Login

- **Pi IP:** `10.42.0.1`  
- **Username:** `ubuntu`  
- **Password:** `ubuntu`

---

## 1) Connect with PuTTY (Windows)

1. Open **PuTTY**.
2. **Host Name (or IP address):** `10.42.0.1`  
   **Port:** `22` • **Connection type:** SSH  
3. Click **Open** → log in:
   ```
   login as: ubuntu
   password: ubuntu
   ```

---

## 2) Ensure a Single ROS Master (free port 11311)

In the PuTTY session on the Pi:

```bash
# See if a master/core is running
ps -ef | grep -E 'rosmaster|roscore'

# Check what's using port 11311
sudo fuser -n tcp 11311

# Kill anything stuck on 11311
sudo fuser -k 11311/tcp
```

---

## 3) Launch the Dobot Driver (ROS1 Noetic)

```bash
roslaunch dobot_magician_driver dobot_magician.launch
```

---
