from threading import Thread
import time
import serial

class bluetooth_thread(Thread):
    def __init__(self):
        Thread.__init__(self)
        self.ser = serial.Serial(port="/dev/ttyS0", baudrate=9600, parity=serial.PARITY_NONE, timeout=0.2)
        self.running = True

    def run(self):
        while self.running:
            x = self.ser.read()
            x = x.decode("utf-8")
            ##            print(x)
            if(x != " "):
            ##                print(x)
                if(x == "c"):
                    print("center")
                    car.full_stop()
                    # time.sleep(.05)
                    
                elif(x == "l"):
                    print("left")
                    car.steer_left()
                    # time.sleep(.05)
                    
                elif(x == "r"):
                    print("right")
                    car.steer_right()
                    # time.sleep(.05)
                    
                elif(x == "f"):
                    print("forward")
                    car.full_forward()
                    # speed = 1600
                    # SetSpeed(speed)
                                
                elif(x == "b"):
                    print("backward")
                    car.full_backward()
                    # speed = 1350
                    # SetSpeed(speed)

                elif(x == "s"):
                    print("stop")
                    car.full_stop()
                    # speed = 1500
                    # SetSpeed(speed)

                elif(x == "X"):
                    print("Print") 
            
            time.sleep(0.1)
    def stop(self):
        self.running = False

##time.sleep(5)
##data.stop()
