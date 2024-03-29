from abc import ABC, abstractmethod
import numpy as np

from .state import Analytic

class ObsModel(ABC):
    def __init__(self, states=[]):
        self.states = states

    @abstractmethod
    def make_measurement(truth, observers, verbose=False):
        pass

class DummyModel(ObsModel):
    def __init__(self):
        super().__init__(states=[])

    def _compute(self, truth, observer):
        return None
    
    def get_available_actions(self, truths, observers, env):
        # available_actions = [0]
        # if env.n_obs < env.budget:
        #     available_actions = [0, 1, 2]

        return np.array([0, 1, 2])


    def make_measurement(self, truth, observers, verbose=False):

        dim = truth.shape[0]
        num_observers = observers.size

        R_invs = np.zeros(shape=(dim, dim, num_observers))
        Z = np.zeros(shape=(dim, num_observers))

        for i in range(3):
            R_invs[i, i, :] = 1/(0.001**2)   # positional uncertaintiy of +- 384 km (VERY CONSERVATIVE, most sensor can do better)
            R_invs[3 + i, 3 + i, :] = 1/(0.01**2) # velocity uncertainty ~ +- 0.01 km/s

        Z_transpose = np.array([truth.x + np.random.multivariate_normal(mean=np.zeros(dim), cov=np.linalg.inv(R_invs[:,:,j])) for j in range(num_observers)])
        Z = Z_transpose.T

        if verbose:
            print(f"Observation Matrix : {Z}")


        return Z, R_invs




class ApparentMag(ObsModel):
    def __init__(self, params, tstep_, sun_phasing = 0):

        AU = 1.496e8 / 384400 # 1 AU (LU)
        ws = - 0.9253018261815922 # angular velocity of sun (rad/TU)

        x0 = np.array([AU * np.cos(sun_phasing), AU * np.sin(sun_phasing), 0])
        tstep = tstep_
        functions = np.array([lambda t : AU*np.cos(ws*t + sun_phasing), lambda t : AU*np.sin(ws*t + sun_phasing), lambda t: 0])

        self.params = params
        sun = Analytic(x0, tstep, functions)
        super().__init__(np.array([sun]))
        
    
    def _compute(self, truth, observer):

        if self._deadzone(truth, observer, body="Earth") or self._deadzone(truth, observer, body="Moon"):
            apmag = np.inf

        else:
            ms = self.params["ms"]
            aspec = self.params["aspec"]
            adiff = self.params["adiff"]
            d = self.params["d"]

            rS = self.states[0].x[:3]
            rO = observer.x[:3]
            rT = truth.x[:3]


            rOT = rT - rO
            rST = rT - rS

            zeta = np.linalg.norm(rOT)
            psi = np.arctan2(np.linalg.norm(np.cross(rOT, rST)), np.dot(rOT, rST))
            pdiff = (2/(3*np.pi)) * (np.sin(psi) + (np.pi - psi)*np.cos(psi))


            apmag = ms - 2.5 * np.log10((d**2)/(zeta**2)*(aspec/4 + adiff*pdiff))

        return apmag
    
    def _deadzone(self, truth, observer, body):
        
        rT = truth.x[:3]
        rO = observer.x[:3]
        if body == "Earth":
            alpha = self.params["rearth"]
            rE = [-self.params["mu"], 0, 0]
        elif body == "Moon":
            alpha = self.params["rmoon"]
            rE = [1- self.params["mu"], 0, 0]
        else:
            ValueError("argument body must be either Earth or Moon")

        rOE = rE - rO

        if rOE[1] == 0. and rOE[2] == 0.:
            w1 = np.array([0, 1, 1])
        else:
            w1 = np.array([0, -rOE[2], rOE[1]])

        w1 =  w1 / np.linalg.norm(w1)
        b1 = np.dot(w1, rE)

        w2 = np.cross(rOE, w1) / np.linalg.norm(rOE)
        b2 = np.dot(w2, rE)

        w3 = rOE / np.linalg.norm(rOE)
        b3 = np.dot(w3, rE)

        if np.abs(np.dot(w1, rT) - b1) <= alpha and np.abs(np.dot(w2, rT)-b2) <= alpha and np.dot(w3, rT) - b3 > 0:
            return True
        else:
            return False
        
    def make_measurement(self, truth, observers, verbose=False):

        apmags = np.zeros(observers.size)

        for i, observer in enumerate(observers):
            apmags[i] = self._compute(truth, observer)

        if verbose:
            print(apmags)

        apmags = np.delete(apmags, np.where(apmags == np.inf))

        if apmags.size == 0:
            Z, R_invs = None, None

        else:
            Z = np.zeros(shape=(6,apmags.size))
            R_invs = np.zeros(shape=(6,6, apmags.size))

            for i, apmag in enumerate(apmags):
                sigma_r = 0.001 * (2 ** (1/3))**(apmag-22)
                sigma_v = 0.01  * (2 ** (1/3))**(apmag-22)

                R_invs[:,:,i] = np.diag([1/sigma_r**2, 1/sigma_r**2, 1/sigma_r**2, 1/sigma_v**2, 1/sigma_v**2, 1/sigma_v**2])

                Z[:,i] = np.array(truth.x + np.random.multivariate_normal(mean=np.zeros(6), cov=np.linalg.inv(R_invs[:,:,i])))
        

        return Z, R_invs
    
    def get_available_actions(self, truths, observers, env):
        available_actions = np.full((observers.size, truths.size + 1), -1, dtype=int)

        for j, observer in enumerate(observers):
        
            for i, truth in enumerate(truths):

                if self.is_visible(truth, observer):
                    available_actions[j, i+1] = i + 1
    
        return available_actions
            

    def is_visible(self, target, observer):
        visible = False
        
        if not (self._deadzone(target, observer, "Moon") or self._deadzone(target, observer, "Earth")):
            visible = True

        return visible
