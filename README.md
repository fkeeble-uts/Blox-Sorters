# Blox-Sorters
Track A Final Project for Sensors and Control for Mechatronic Systems. Automatic Lego colour sorting system using a DoBot robotic arm and an RGB-D Sensor. 


Instructions to make Git Commits are as follows 

1. Clone Repository with link on github
2. Configure your identity using:
git config --global user.email "<your-github-email>"
git config --global user.name "<your-github-username>"

3. Stage all new changes to Git using:
git add . 

4. Commit changes locally using:
git commit -m "<add your commit message here (helpful to know what you changed)>"

5. Push to github using:
git push origin main


# Start the Dobot Magician via PuTTY (Raspberry Pi)

This guide shows how to SSH into your Raspberry Pi with **PuTTY**

## Network & Login

- **Pi IP:** `10.42.0.1`  
- **Username:** `ubuntu`  
- **Password:** `ubuntu`

## 1) Connect with PuTTY (Windows)

1. Open **PuTTY**.
2. **Host Name (or IP address):** `10.42.0.1`  
   **Port:** `22` • **Connection type:** SSH  
3. Click **Open** → log in:
   ```
   login as: ubuntu
   password: ubuntu
   ```

## 3) Launch the Dobot Driver (ROS1 Noetic)

```bash
source catkin_ws/devel/setup.bash
export ROS_HOSTNAME=10.42.0.1
export ROS_MASTER_URI=http://10.42.0.1:11311
export ROS_IP=10.42.0.1
roslaunch dobot_magician_driver dobot_magician.launch 
```

