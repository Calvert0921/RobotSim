import sim
import time
import sys
import cv2
import numpy as np

#关闭之前的连接
sim.simxFinish(-1)

# 获得客户端ID
clientID = sim.simxStart('127.0.0.1',19995,True,True,5000,5)
print("Connection success!!!")

if clientID != -1:
    print('Connected to remote API server')
else:
    print('Connection not successful')
    sys.exit('Could not connect')

# 启动仿真
sim.simxStartSimulation(clientID,sim.simx_opmode_blocking)
print("Simulation start")

# 使能同步模式
sim.simxSynchronous(clientID,True)

# 获得对象的句柄
ret, targetObj = sim.simxGetObjectHandle(clientID,'target',sim.simx_opmode_blocking)
errorCode,visionSensorHandle = sim.simxGetObjectHandle(clientID,'visionSensor',sim.simx_opmode_oneshot_wait)
errprCode,resolution,rawimage = sim.simxGetVisionSensorImage(clientID,visionSensorHandle,0,sim.simx_opmode_streaming)



def readVisionSensor():
    global resolution
    err_code, resolution, rawimage = sim.simxGetVisionSensorImage(clientID, visionSensorHandle, 0, sim.simx_opmode_buffer)
    if err_code != sim.simx_return_ok or rawimage == []:
        print("No image yet...")
        return None
    # fix: convert signed int8 to uint8
    sensorImage = np.array(rawimage, dtype=np.int8).astype(np.uint8)
    sensorImage = sensorImage.reshape((resolution[1], resolution[0], 3))
    sensorImage = cv2.flip(sensorImage, 0)
    sensorImage = cv2.cvtColor(sensorImage, cv2.COLOR_RGB2BGR)
    return sensorImage

def readDepthSensor():
    global resolution
    # 获取 Depth Info
    sim_ret, resolution, depth_buffer = sim.simxGetVisionSensorDepthBuffer(clientID, visionSensorHandle, sim.simx_opmode_blocking)
    depth_img = np.asarray(depth_buffer)
    depth_img.shape = (resolution[1], resolution[0])
    zNear = 0.01
    zFar = 2
    depth_img = depth_img * (zFar - zNear) + zNear
    depth_img = cv2.flip(depth_img, 0)
    return depth_img

while True:
    # 获得对象的位置，并输出
    ret, arr = sim.simxGetObjectPosition(clientID,targetObj,-1,sim.simx_opmode_blocking)

    image = readVisionSensor()
    # depth = readDepthSensor()
    # print(depth)
    cv2.imshow("image", image)
    # cv2.imshow("depth", depth)
    cv2.waitKey(1)
    # saveFile = ".\image.jpg"  # 保存文件的路径
    # cv2.imwrite(saveFile, depth)  # 保存图像文件


    if ret == sim.simx_return_ok:
        print(arr)

    # time.sleep(2)

# 退出
sim.simxFinish(clientID)
print('Program end')
