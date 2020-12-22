import rclpy

from std_srvs.srv._empty import Empty_Request
from std_srvs.srv import Empty

class GazeboConnection():
    
    def __init__(self, node):
        self.node = node
        
        self.pause = self.node.create_client(Empty, '/pause_physics')
        self.unpause = self.node.create_client(Empty, '/unpause_physics')
        self.reset_proxy = self.node.create_client(Empty, '/reset_simulation')
    
    def pauseSim(self):
        self.pause.call(Empty_Request())
        
    def unpauseSim(self):
        self.unpause.call(Empty_Request())
        
    def resetSim(self):
        self.reset_proxy.call(Empty_Request())