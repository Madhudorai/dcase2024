import numpy as np
'''
MicrophoneArray: This script contains Microphone, MicrophoneArray, EigenmikeEM32 classes
'''


class Microphone(object):
    """
    Microphone class
    """

    def __init__(self, name='Generic', version='1.0', direct='Omnidirectional'):
        """
        Constructor

        :param name: Name of the microphone
        :param direct: Directivity of the microphone (str)
        """
        self._micname = name
        self._ver = version
        self._directivity = direct

    def getname(self):
        """
        Get the name

        :return: Name (str) of the microphone object
        """
        return self._micname

    def setname(self, name):
        """
        Set for the name attribute

        :param name:  Name of the microphone (str)
        """
        self._micname = name

    def getversion(self):
        """
        Get the version

        :return: Version(str) of the microphone object
        """
        print(self._ver)

    def setversion(self, version):
        """
        Set for the version

        :param version: Version of the microphone (str)
        """
        self._ver = version

class MicrophoneArray(Microphone):
    """
    MicrophoneArray class inherits from Microphone class
    """

    def __init__(self, name, typ, version, direct):
        """
        Constructor

        :param name: Name of the array
        :param typ: Type of the array
        :param version: Version of the array
        :param direct: Directivity of components
        """
        super(MicrophoneArray, self).__init__(name, version, direct)
        self._arraytype = None
        self.__arraytype = typ

    def gettype(self):
        """
        Get for the array type

        :return:
        """
        return self.__arraytype

    def settype(self, typ):
        """
        set for array type

        :param typ: type of the array (str)
        """
        self.__arraytype = typ


class EigenmikeEM32(MicrophoneArray):
    """
    Eigenmike em32 class inherits from the MicrophoneArray class
    """
    def __init__(self):
        super(EigenmikeEM32, self).__init__('Eigenmike 32', 'Rigid Spherical', '17.0', 'Omni')
        self._numelements = 32

        self._thetas = np.array([69.0, 90.0, 111.0, 90.0, 32.0, 55.0,
                                 90.0, 125.0, 148.0, 125.0, 90.0, 55.0, 21.0, 58.0,
                                 121.0, 159.0, 69.0, 90.0, 111.0, 90.0, 32.0, 55.0,
                                 90.0, 125.0, 148.0, 125.0, 90.0, 55.0, 21.0, 58.0,
                                 122.0, 159.0]) / 180.0 * np.pi

        self._phis = np.array([0.0, 32.0, 0.0, 328.0, 0.0, 45.0, 69.0, 45.0, 0.0, 315.0,
                               291.0, 315.0, 91.0, 90.0, 90.0, 89.0, 180.0, 212.0, 180.0, 148.0, 180.0,
                               225.0, 249.0, 225.0, 180.0, 135.0, 111.0, 135.0, 269.0, 270.0, 270.0,
                               271.0]) / 180.0 * np.pi

        self._radius = 4.2e-2

        self._weights = np.ones(32)

    def returnArrayStruct(self):
        """
        Returns the attributes of the Eigenmike em32 as a struct

        :param self:
        :return: dict object with the name, type, thetas, phis, radius, weights, numelements, directivity
        """
        em32 = {'name': self._micname,
                'thetas': self._thetas,
                'phis': self._phis,
                'radius': self._radius,
                'weights': self._weights,
                'version': self._weights,
                'numelements': self._numelements,
                'directivity': self._directivity}
        return em32



