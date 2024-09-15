from typing import Any, Dict, Optional, Tuple, Union


import chex
from flax import struct
import jax
from jax import lax
import jax.numpy as jnp
from gymnax.environments import environment
from gymnax.environments import spaces
from jax import random
import numpy as np
from typing import Any, Dict, Tuple, Union
import math


REF_W = 24 * 2
REF_H = REF_W
REF_U = 1.5  # ground height
REF_WALL_WIDTH = 1.0  # wall width
REF_WALL_HEIGHT = 3.5
PLAYER_SPEED_X = 10 * 1.75
PLAYER_SPEED_Y = 10 * 1.35
MAX_BALL_SPEED = 15 * 1.5
TIMESTEP = 1 / 30.0
NUDGE = 0.1
FRICTION = 1.0  # 1 means no FRICTION, less means FRICTION
GRAVITY = -9.8 * 2 * 1.5

MAXLIVES = 5  # game ends when one agent loses this many games

WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 500

FACTOR = WINDOW_WIDTH / REF_W

# if set to true, renders using cv2 directly on numpy array
# (otherwise uses pyglet / opengl -> much smoother for human player)
PIXEL_MODE = True
PIXEL_SCALE = 2  # Render at multiple of Pixel Obs resolution, then downscale.

PIXEL_WIDTH = 84 * 2 * 2
PIXEL_HEIGHT = 84 * 2


@struct.dataclass
class ParticleState(object):
    x: jnp.float32
    y: jnp.float32
    prev_x: jnp.float32
    prev_y: jnp.float32
    vx: jnp.float32
    vy: jnp.float32
    r: jnp.float32


@struct.dataclass
class AgentState(object):
    direction: jnp.int32  # -1 means left, 1 means right player.
    x: jnp.float32
    y: jnp.float32
    r: jnp.float32
    vx: jnp.float32
    vy: jnp.float32
    desired_vx: jnp.float32
    desired_vy: jnp.float32
    life: jnp.int32


@struct.dataclass
class GameState(object):
    ball: ParticleState
    agent_left: AgentState
    agent_right: AgentState
    hidden_left: jnp.ndarray  # rnn hidden state for internal policy
    hidden_right: jnp.ndarray
    action_left_flag: jnp.int32  # if 1, then use the action action_left
    action_left: jnp.ndarray
    action_right_flag: jnp.int32  # same as above
    action_right: jnp.ndarray


@struct.dataclass
class EnvState(environment.EnvState):
    game_state: GameState
    obs: jnp.ndarray
    time: int
    key: jnp.ndarray


@struct.dataclass
class EnvParams(environment.EnvParams):
    max_steps_in_episode: int = 3000


@struct.dataclass
class BaselinePolicyParams(object):
    w: jnp.ndarray
    b: jnp.ndarray


def initBaselinePolicyParams():
    nGameInput = 8  # 8 states for agent
    nGameOutput = 3  # 3 buttons (forward, backward, jump)
    nRecurrentState = 4  # extra recurrent states for feedback.
    """See training details:
    https://blog.otoro.net/2015/03/28/neural-slime-volleyball/
    """
    weight = jnp.array(
        [
            7.5719,
            4.4285,
            2.2716,
            -0.3598,
            -7.8189,
            -2.5422,
            -3.2034,
            0.3935,
            1.2202,
            -0.49,
            -0.0316,
            0.5221,
            0.7026,
            0.4179,
            -2.1689,
            1.646,
            -13.3639,
            1.5151,
            1.1175,
            -5.3561,
            5.0442,
            0.8451,
            0.3987,
            -2.9501,
            -3.7811,
            -5.8994,
            6.4167,
            2.5014,
            7.338,
            -2.9887,
            2.4586,
            13.4191,
            2.7395,
            -3.9708,
            1.6548,
            -2.7554,
            -1.5345,
            -6.4708,
            9.2426,
            -0.7392,
            0.4452,
            1.8828,
            -2.6277,
            -10.851,
            -3.2353,
            -4.4653,
            -3.1153,
            -1.3707,
            7.318,
            16.0902,
            1.4686,
            7.0391,
            1.7765,
            -1.155,
            2.6697,
            -8.8877,
            1.1958,
            -3.2839,
            -5.4425,
            1.6809,
            7.6812,
            -2.4732,
            1.738,
            0.3781,
            0.8718,
            2.5886,
            1.6911,
            1.2953,
            -9.0052,
            -4.6038,
            -6.7447,
            -2.5528,
            0.4391,
            -4.9278,
            -3.6695,
            -4.8673,
            -1.6035,
            1.5011,
            -5.6124,
            4.9747,
            1.8998,
            3.0359,
            6.2983,
            -4.8568,
            -2.1888,
            -4.1143,
            -3.9874,
            -0.0459,
            4.7134,
            2.8952,
            -9.3627,
            -4.685,
            0.3601,
            -1.3699,
            9.7294,
            11.5596,
            0.1918,
            3.0783,
            0.0329,
            -0.1362,
            -0.1188,
            -0.7579,
            0.3278,
            -0.977,
            -0.9377,
        ]
    )
    weight = weight.reshape(
        nGameOutput + nRecurrentState, nGameInput + nGameOutput + nRecurrentState
    )
    bias = jnp.array([2.2935, -2.0353, -1.7786, 5.4567, -3.6368, 3.4996, -0.0685])

    return BaselinePolicyParams(weight, bias)


class Wall:
    """used for the fence, and also the ground"""

    def __init__(self, x, y, w, h, c):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.c = c


def initParticleState(x, y, vx, vy, r):
    return ParticleState(
        jnp.float32(x),
        jnp.float32(y),
        jnp.float32(x),
        jnp.float32(y),
        jnp.float32(vx),
        jnp.float32(vy),
        jnp.float32(r),
    )


class Particle:
    """used for the ball, and also for the round stub above the fence"""

    def __init__(self, p: ParticleState, c):
        self.p = p
        self.c = c

    def move(self):
        self.p = ParticleState(
            self.p.x + self.p.vx * TIMESTEP,
            self.p.y + self.p.vy * TIMESTEP,
            self.p.x,
            self.p.y,
            self.p.vx,
            self.p.vy,
            r=self.p.r,
        )

    def applyAcceleration(self, ax, ay):
        self.p = ParticleState(
            self.p.x,
            self.p.y,
            self.p.prev_x,
            self.p.prev_y,
            self.p.vx + ax * TIMESTEP,
            self.p.vy + ay * TIMESTEP,
            r=self.p.r,
        )

    def checkEdges(self):
        oldp = self.p
        return_sign = jnp.where(oldp.x <= 0, -1, 1)
        newx = oldp.x
        newy = oldp.y
        newpx = oldp.prev_x
        newpy = oldp.prev_y
        newvx = oldp.vx
        newvy = oldp.vy

        newx = jnp.where(
            oldp.x <= (oldp.r - REF_W / 2), oldp.r - REF_W / 2 + NUDGE * TIMESTEP, newx
        )
        newvx = jnp.where(oldp.x <= (oldp.r - REF_W / 2), oldp.vx * (-FRICTION), newvx)

        newx = jnp.where(
            oldp.x >= (REF_W / 2 - oldp.r), REF_W / 2 - oldp.r - NUDGE * TIMESTEP, newx
        )
        newvx = jnp.where(oldp.x >= (REF_W / 2 - oldp.r), oldp.vx * (-FRICTION), newvx)

        return_value = jnp.where(oldp.y <= (oldp.r + REF_U), 1, 0)

        newy = jnp.where(
            oldp.y <= (oldp.r + REF_U), oldp.r + REF_U + NUDGE * TIMESTEP, newy
        )
        newvy = jnp.where(oldp.y <= (oldp.r + REF_U), oldp.vy * (-FRICTION), newvy)

        newy = jnp.where(
            oldp.y >= (REF_H - oldp.r), REF_H - oldp.r - NUDGE * TIMESTEP, newy
        )

        newvy = jnp.where(oldp.y >= (REF_H - oldp.r), oldp.vy * (-FRICTION), newvy)

        # fence:

        newx = jnp.where(
            (oldp.x <= (REF_WALL_WIDTH / 2 + oldp.r))
            & (oldp.prev_x > (REF_WALL_WIDTH / 2 + oldp.r))
            & (oldp.y <= REF_WALL_HEIGHT),
            REF_WALL_WIDTH / 2 + oldp.r + NUDGE * TIMESTEP,
            newx,
        )
        newvx = jnp.where(
            (oldp.x <= (REF_WALL_WIDTH / 2 + oldp.r))
            & (oldp.prev_x > (REF_WALL_WIDTH / 2 + oldp.r))
            & (oldp.y <= REF_WALL_HEIGHT),
            oldp.vx * (-FRICTION),
            newvx,
        )

        newx = jnp.where(
            (oldp.x >= (-REF_WALL_WIDTH / 2 - oldp.r))
            & (oldp.prev_x < (-REF_WALL_WIDTH / 2 - oldp.r))
            & (oldp.y <= REF_WALL_HEIGHT),
            -REF_WALL_WIDTH / 2 - oldp.r - NUDGE * TIMESTEP,
            newx,
        )
        newvx = jnp.where(
            (oldp.x >= (-REF_WALL_WIDTH / 2 - oldp.r))
            & (oldp.prev_x < (-REF_WALL_WIDTH / 2 - oldp.r))
            & (oldp.y <= REF_WALL_HEIGHT),
            oldp.vx * (-FRICTION),
            newvx,
        )

        self.p = ParticleState(newx, newy, newpx, newpy, newvx, newvy, oldp.r)
        return return_value * return_sign

    def bounce(self, p):  # bounce two balls that have collided (this and that)
        oldp = self.p
        abx = oldp.x - p.x
        aby = oldp.y - p.y
        abd = jnp.sqrt(abx * abx + aby * aby)
        abx /= abd  # normalize
        aby /= abd
        nx = abx  # reuse calculation
        ny = aby
        abx *= NUDGE
        aby *= NUDGE

        new_y = oldp.y
        new_x = oldp.x
        dy = new_y - p.x
        dx = new_x - p.y

        total_r = oldp.r + p.r
        total_r2 = total_r * total_r

        # this was a while loop in the orig code, but most cases < 15.
        for i in range(15):
            total_d2 = dy * dy + dx * dx
            new_x = jnp.where(total_d2 < total_r2, new_x + abx, new_x)
            new_y = jnp.where(total_d2 < total_r2, new_y + aby, new_y)
            dy = p.y - new_y
            dx = p.x - new_x

        ux = oldp.vx - p.vx
        uy = oldp.vy - p.vy
        un = ux * nx + uy * ny
        unx = nx * (un * 2.0)  # added factor of 2
        uny = ny * (un * 2.0)  # added factor of 2
        ux -= unx
        uy -= uny
        return ParticleState(
            x=new_x,
            y=new_y,
            prev_x=oldp.prev_x,
            prev_y=oldp.prev_y,
            vx=ux + p.vx,
            vy=uy + p.vy,
            r=oldp.r,
        )

    def bounceIfColliding(self, p):
        dy = p.y - self.p.y
        dx = p.x - self.p.x
        d2 = dx * dx + dy * dy
        r = self.p.r + p.r
        r2 = r * r
        newp = self.bounce(p)

        # make if condition work with jax:
        newx = jnp.where(d2 < r2, newp.x, self.p.x)
        newy = jnp.where(d2 < r2, newp.y, self.p.y)
        newprev_x = jnp.where(d2 < r2, newp.prev_x, self.p.prev_x)
        newprev_y = jnp.where(d2 < r2, newp.prev_y, self.p.prev_y)
        newvx = jnp.where(d2 < r2, newp.vx, self.p.vx)
        newvy = jnp.where(d2 < r2, newp.vy, self.p.vy)
        self.p = ParticleState(
            x=newx,
            y=newy,
            prev_x=newprev_x,
            prev_y=newprev_y,
            vx=newvx,
            vy=newvy,
            r=self.p.r,
        )

    def limitSpeed(self, maxSpeed):
        oldp = self.p
        mag2 = oldp.vx * oldp.vx + oldp.vy * oldp.vy
        mag = jnp.sqrt(mag2)

        newvx = oldp.vx
        newvy = oldp.vy
        newvx = jnp.where(mag2 > (maxSpeed * maxSpeed), newvx / mag, newvx)
        newvy = jnp.where(mag2 > (maxSpeed * maxSpeed), newvy / mag, newvy)
        newvx = jnp.where(mag2 > (maxSpeed * maxSpeed), newvx * maxSpeed, newvx)
        newvy = jnp.where(mag2 > (maxSpeed * maxSpeed), newvy * maxSpeed, newvy)

        self.p = ParticleState(
            x=oldp.x,
            y=oldp.y,
            prev_x=oldp.prev_x,
            prev_y=oldp.prev_y,
            vx=newvx,
            vy=newvy,
            r=oldp.r,
        )


@struct.dataclass
class Observation(object):  # is also the "RelativeState" in the original code
    """
    keeps track of the obs.
    Note: the observation is from the perspective of the agent.
    an agent playing either side of the fence must see obs the same way
    """

    x: jnp.float32  # agent
    y: jnp.float32
    vx: jnp.float32
    vy: jnp.float32
    bx: jnp.float32  # ball
    by: jnp.float32
    bvx: jnp.float32
    bvy: jnp.float32
    ox: jnp.float32  # opponent
    oy: jnp.float32
    ovx: jnp.float32
    ovy: jnp.float32


def getZeroObs() -> Observation:
    return Observation(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)


def getObsArray(rs: Observation):
    # scale inputs to be in the order
    # of magnitude of 10 for neural network. (legacy)
    scaleFactor = 10.0
    result = (
        jnp.array(
            [
                rs.x,
                rs.y,
                rs.vx,
                rs.vy,
                rs.bx,
                rs.by,
                rs.bvx,
                rs.bvy,
                rs.ox,
                rs.oy,
                rs.ovx,
                rs.ovy,
            ]
        )
        / scaleFactor
    )

    return result


class Agent:
    """keeps track of the agent in the game. note: not the policy network"""

    def __init__(self, agent, c):
        self.p = agent
        self.state = getZeroObs()
        self.c = c

    def setAction(self, action):
        forward = jnp.int32(0)
        backward = jnp.int32(0)
        jump = jnp.int32(0)
        if len(action.shape) == 1:
            forward = jnp.where(action[0] > 0, 1, forward)
            backward = jnp.where(action[1] > 0, 1, backward)
            jump = jnp.where(action[2] > 0, 1, jump)
        else:
            forward = jnp.where(action[:, 0] > 0, 1, forward)
            backward = jnp.where(action[:, 1] > 0, 1, backward)
            jump = jnp.where(action[:, 2] > 0, 1, jump)
        new_desired_vx = jnp.float32(0.0)
        new_desired_vy = jnp.float32(0.0)

        new_desired_vx = jnp.where(
            forward & (1 - backward), -PLAYER_SPEED_X, new_desired_vx
        )
        new_desired_vx = jnp.where(
            backward & (1 - forward), PLAYER_SPEED_X, new_desired_vx
        )
        new_desired_vy = jnp.where(jump, PLAYER_SPEED_Y, new_desired_vy)

        p = self.p
        self.p = AgentState(
            p.direction,
            p.x,
            p.y,
            p.r,
            p.vx,
            p.vy,
            new_desired_vx,
            new_desired_vy,
            p.life,
        )

    def move(self):
        p = self.p
        new_x = p.x + p.vx * TIMESTEP
        self.p = AgentState(
            p.direction,
            new_x,
            p.y + p.vy * TIMESTEP,
            p.r,
            p.vx,
            p.vy,
            p.desired_vx,
            p.desired_vy,
            p.life,
        )

    def update(self):
        p = self.p
        new_vy = p.vy + GRAVITY * TIMESTEP

        new_vy = jnp.where(p.y <= REF_U + NUDGE * TIMESTEP, p.desired_vy, new_vy)
        new_vx = p.desired_vx * p.direction
        self.p = AgentState(
            p.direction,
            p.x,
            p.y,
            p.r,
            new_vx,
            new_vy,
            p.desired_vx,
            p.desired_vy,
            p.life,
        )

        self.move()

        p = self.p

        # stay in their own half:

        new_y = p.y
        new_vy = p.vy
        new_y = jnp.where(p.y <= REF_U, REF_U, new_y)
        new_vy = jnp.where(p.y <= REF_U, 0, new_vy)

        # stay in their own half:
        new_vx = p.vx
        new_x = p.x

        new_vx = jnp.where(p.x * p.direction <= (REF_WALL_WIDTH / 2 + p.r), 0, new_vx)
        new_x = jnp.where(
            p.x * p.direction <= (REF_WALL_WIDTH / 2 + p.r),
            p.direction * (REF_WALL_WIDTH / 2 + p.r),
            new_x,
        )

        new_vx = jnp.where(p.x * p.direction >= (REF_W / 2 - p.r), 0, new_vx)
        new_x = jnp.where(
            p.x * p.direction >= (REF_W / 2 - p.r),
            p.direction * (REF_W / 2 - p.r),
            new_x,
        )

        self.p = AgentState(
            p.direction,
            new_x,
            new_y,
            p.r,
            new_vx,
            new_vy,
            p.desired_vx,
            p.desired_vy,
            p.life,
        )

    def updateLife(self, result):
        """updates the life based on result and internal direction"""
        p = self.p
        updateAmount = p.direction * result  # only update if this value is -1
        new_life = jnp.where(updateAmount < 0, p.life - 1, p.life)
        self.p = AgentState(
            p.direction, p.x, p.y, p.r, p.vx, p.vy, p.desired_vx, p.desired_vy, new_life
        )

    def updateState(self, ball: ParticleState, opponent: AgentState):
        """normalized to side, customized for each agent's perspective"""
        p = self.p
        # agent's self
        x = p.x * p.direction
        y = p.y
        vx = p.vx * p.direction
        vy = p.vy
        # ball
        bx = ball.x * p.direction
        by = ball.y
        bvx = ball.vx * p.direction
        bvy = ball.vy
        # opponent
        ox = opponent.x * (-p.direction)
        oy = opponent.y
        ovx = opponent.vx * (-p.direction)
        ovy = opponent.vy

        self.state = Observation(x, y, vx, vy, bx, by, bvx, bvy, ox, oy, ovx, ovy)

    def getObservation(self):
        return getObsArray(self.state)


def baselinePolicy(obs: jnp.ndarray, state: jnp.ndarray, params: BaselinePolicyParams):
    """take obs, prev rnn state, return updated rnn state, action"""
    nGameInput = 8  # 8 states that policy cares about (ignores last 4)
    nGameOutput = 3  # 3 buttons (forward, backward, jump)

    weight = params.w
    bias = params.b
    inputState = jnp.concatenate([obs[:nGameInput], state])
    outputState = jnp.tanh(jnp.dot(weight, inputState) + bias)
    action = jnp.zeros(nGameOutput)
    action = jnp.where(outputState[:nGameOutput] > 0.75, 1, action)
    return outputState, action


class Game:
    """
    the main slime volley game.
    can be used in various settings,
    such as ai vs ai, ai vs human, human vs human
    """

    def __init__(self, gameState):
        self.baselineParams = initBaselinePolicyParams()
        self.ground = None
        self.fence = None
        self.fenceStub = None
        self.reset(gameState)

    def reset(self, gameState):
        self.ground = Wall(0, 0.75, REF_W, REF_U)
        self.fence = Wall(
            0,
            0.75 + REF_WALL_HEIGHT / 2,
            REF_WALL_WIDTH,
            (REF_WALL_HEIGHT - 1.5),
        )
        fenceStubParticle = initParticleState(
            0, REF_WALL_HEIGHT, 0, 0, REF_WALL_WIDTH / 2
        )
        self.fenceStub = Particle(fenceStubParticle)
        self.setGameState(gameState)

    def setGameState(self, gameState):
        self.ball = Particle(gameState.ball)
        self.agent_left = Agent(
            gameState.agent_left,
        )
        self.agent_right = Agent(
            gameState.agent_right,
        )
        self.agent_left.updateState(self.ball.p, self.agent_right.p)
        self.agent_right.updateState(self.ball.p, self.agent_left.p)
        self.hidden_left = gameState.hidden_left
        self.hidden_right = gameState.hidden_right
        self.action_left_flag = gameState.action_left_flag
        self.action_left = gameState.action_left
        self.action_right_flag = gameState.action_right_flag
        self.action_right = gameState.action_right

    def setLeftAction(self, action):
        self.action_left_flag = jnp.int32(1)
        self.action_left = action

    def setRightAction(self, action):
        self.action_right_flag = jnp.int32(1)
        self.action_right = action

    def setAction(self):
        obs_left = self.agent_left.getObservation()
        obs_right = self.agent_right.getObservation()
        self.hidden_left, action_left = baselinePolicy(
            obs_left, self.hidden_left, self.baselineParams
        )
        self.hidden_right, action_right = baselinePolicy(
            obs_right, self.hidden_right, self.baselineParams
        )
        # overwrite internal AI actions if the flags are turned on:
        action_left = jnp.where(self.action_left_flag, self.action_left, action_left)
        action_right = jnp.where(
            self.action_right_flag, self.action_right, action_right
        )
        self.agent_left.setAction(action_left)
        self.agent_right.setAction(action_right)

    def getGameState(self):
        return GameState(
            self.ball.p,
            self.agent_left.p,
            self.agent_right.p,
            self.hidden_left,
            self.hidden_right,
            self.action_left_flag,
            self.action_left,
            self.action_right_flag,
            self.action_right,
        )

    def step(self):
        """main game loop"""
        self.agent_left.update()
        self.agent_right.update()
        self.ball.applyAcceleration(0, GRAVITY)
        self.ball.limitSpeed(MAX_BALL_SPEED)
        self.ball.move()

        self.ball.bounceIfColliding(self.agent_left.p)
        self.ball.bounceIfColliding(self.agent_right.p)
        self.ball.bounceIfColliding(self.fenceStub.p)

        # negated, since we want reward to be from the persepctive
        # of right agent being trained.
        result = -self.ball.checkEdges()

        self.agent_left.updateLife(result)
        self.agent_right.updateLife(result)

        self.agent_left.updateState(self.ball.p, self.agent_right.p)
        self.agent_right.updateState(self.ball.p, self.agent_left.p)

        return result


def get_random_ball_v(key: jnp.ndarray):
    result = random.uniform(key, shape=(2,)) * 2 - 1
    ball_vx = result[1] * 20
    ball_vy = result[2] * 7.5 + 17.5
    return ball_vx, ball_vy


def update_state_for_new_match(game_state: GameState, reward, key: jnp.ndarray):
    old_ball = game_state.ball
    ball_vx, ball_vy = get_random_ball_v(key)
    new_ball = initParticleState(0, REF_W / 4, ball_vx, ball_vy, 0.5)
    x = jnp.where(reward == 0, old_ball.x, new_ball.x)
    y = jnp.where(reward == 0, old_ball.y, new_ball.y)
    prev_x = jnp.where(reward == 0, old_ball.prev_x, new_ball.prev_x)
    prev_y = jnp.where(reward == 0, old_ball.prev_y, new_ball.prev_y)
    vx = jnp.where(reward == 0, old_ball.vx, new_ball.vx)
    vy = jnp.where(reward == 0, old_ball.vy, new_ball.vy)
    ball = ParticleState(x, y, prev_x, prev_y, vx, vy, old_ball.r)
    p = game_state
    return GameState(
        ball,
        p.agent_left,
        p.agent_right,
        p.hidden_left,
        p.hidden_right,
        p.action_left_flag,
        p.action_left,
        p.action_right_flag,
        p.action_right,
    )


def update_state(action: jnp.ndarray, game_state: GameState, key: jnp.array):
    game = Game(game_state)
    game.setRightAction(action)
    game.setAction()
    reward = game.step()  # from perspective of the agent on the right
    updated_game_state = game.getGameState()
    obs = game.agent_right.getObservation()

    updated_game_state = update_state_for_new_match(updated_game_state, reward, key)

    return updated_game_state, reward, obs


def initAgentState(direction, x, y):
    return AgentState(direction, x, y, 1.5, 0, 0, 0, 0, MAXLIVES)


def initBaselinePolicyState():
    return jnp.zeros(7)


def get_obs(game_state: GameState):
    game = Game(game_state)
    return game.agent_right.getObservation()


def initGameState(ball_vx, ball_vy):
    ball = initParticleState(0, REF_W / 4, ball_vx, ball_vy, 0.5)
    agent_left = initAgentState(-1, -REF_W / 4, 1.5)
    agent_right = initAgentState(1, REF_W / 4, 1.5)
    hidden_left = initBaselinePolicyState()
    hidden_right = initBaselinePolicyState()
    action_left_flag = jnp.int32(0)  # left is the built-in AI
    action_left = jnp.array([0, 0, 1], dtype=jnp.float32)
    action_right_flag = jnp.int32(1)  # right is the agent being trained.
    action_right = jnp.array([0, 0, 1], dtype=jnp.float32)
    return GameState(
        ball,
        agent_left,
        agent_right,
        hidden_left,
        hidden_right,
        action_left_flag,
        action_left,
        action_right_flag,
        action_right,
    )


def get_init_game_state_fn(key: jnp.ndarray):
    ball_vx, ball_vy = get_random_ball_v(key)
    return initGameState(ball_vx, ball_vy)


class SlimeVolley(environment.Environment[EnvState, EnvParams]):

    def __init__(self):
        super().__init__()
        self.obs_shape = (12,)
        self.action_set = jnp.array([0, 1, 2])

    @property
    def default_params(self) -> EnvParams:
        # Default environment parameters
        return EnvParams()

    def step_env(
        self,
        key: chex.PRNGKey,
        state: EnvState,
        action: Union[int, float, chex.Array],
        params: EnvParams,
    ) -> Tuple[chex.Array, EnvState, jnp.ndarray, jnp.ndarray, Dict[Any, Any]]:
        """Perform single timestep state transition."""
        next_key, key = random.split(state.key)
        # Convert scalar action to one-hot vector
        action = jax.nn.one_hot(action, 3)
        cur_state, reward, obs = update_state(
            action=action, game_state=state.game_state, key=key
        )
        time = state.time + 1
        state = state.replace(time=time)
        done = state.time >= params.max_steps_in_episode
        state = state.replace(game_state=cur_state, obs=obs, time=time, key=next_key)
        return (
            lax.stop_gradient(self.get_obs(state)),
            lax.stop_gradient(state),
            reward.astype(jnp.float32),
            done,
            {},
        )

    def reset_env(
        self, key: chex.PRNGKey, params: EnvParams
    ) -> Tuple[chex.Array, EnvState]:
        next_key, key = random.split(key)
        game_state = get_init_game_state_fn(key)
        state = EnvState(
            game_state=game_state,
            obs=get_obs(game_state),
            time=0,
            key=next_key,
        )
        return self.get_obs(state), state

    def get_obs(self, state: EnvState, params=None, key=None) -> chex.Array:
        """Return observation from raw state trafo."""
        return state.obs

    @property
    def name(self) -> str:
        """Environment name."""
        return "SlimeVolley"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return len(self.action_set)

    def action_space(self, params: Optional[EnvParams] = None) -> spaces.Discrete:
        """Action space of the environment."""
        return spaces.Discrete(len(self.action_set))

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment."""
        return spaces.Box(0, 1, self.obs_shape)
