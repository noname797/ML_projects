from operator import mod
import numpy as np
import pandas as pd

import tensorflow

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense,Input
from tensorflow.keras.optimizers import Adam

from datetime import datetime
import itertools
import argparse
import re
import os
import pickle

from sklearn.preprocessing import StandardScaler

def get_data():
    '''Will return a Tx3 list of stock prices. where each colum is a different stock'''
    
    df=pd.read_csv("aapl_msi_sbux.csv")
    
    return df.values

class ReplayBuffer:
    def __init__(self,obs_dim,act_dim,size):
        self.obs1_buf=np.zeros([size,obs_dim],dtype=np.float32)
        self.obs2_buf=np.zeros([size,obs_dim],dtype=np.float32)
        self.acts_buf=np.zeros(size,dtype=np.uint8) # actions from 0,26
        self.rews_buf=np.zeros(size,dtype=np.float32)
        self.done_buf=np.zeros(size,dtype=np.uint8) # 0,1 hence can be uint8
        self.ptr,self.size,self.max_size=0,0,size
        
    def store(self,obs,act,rew,next_obs,done):
        self.obs1_buf[self.ptr]=obs
        self.obs2_buf[self.ptr]=next_obs
        self.acts_buf[self.ptr]=act
        self.rews_buf[self.ptr]=rew
        self.done_buf[self.ptr]=done
        self.ptr=(self.ptr+1)%self.max_size
        self.size=min(self.size+1,self.max_size)
        
    def sample_batch(self,batch_size=32):
        idxs=np.random.randint(0,self.size,size=batch_size)
        return dict(s=self.obs1_buf[idxs],s2=self.obs2_buf[idxs],
                    a=self.acts_buf[idxs],r=self.rews_buf[idxs],
                    d=self.done_buf[idxs])
    
def get_scaler(env):
    '''Returns scikit learn scaler object to scale the states.
    We can also populate the replay buffer.
    We must have some data, so we play an episode randomly and store each of the states we encounter. There is no need
    have an agent as it wont be trained anyway. Hence we are just sampling a value from action space.
    '''
    states=[]
    for _ in range(env.n_step):
        action=np.random.choice(env.action_space)
        state,reward,done,info=env.step(action)
        states.append(state)
        if done:
            break
    scaler=StandardScaler()
    scaler.fit(states)
    return scaler

    # For making is more approriate we can run if for multiple episodes

def make_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    
def mlp(input_dim,n_action,n_hidden_layers=1,hidden_dim=32):
    i=Input(shape=(input_dim,))
    x=i
    
    for _ in range(n_hidden_layers):
        x=Dense(hidden_dim, activation='relu')(x)
        
    x=Dense(n_action)(x)
    
    model=Model(i,x)
    
    model.compile(loss='mse',optimizer='adam')
    print(model.summary())
    return model

class MultiStockEnv:
    """
    State vector size of 7
    * # of shares of each,price of each,cash owned.
    action vector with 27 (3^3) possibilities
    """
    def __init__(self,data,initial_investment=20000):
        self.stock_price_history=data
        self.n_step,self.n_stock=self.stock_price_history.shape

        self.initial_investment=initial_investment
        self.cur_step=None
        self.stock_owned=None
        self.stock_price=None
        self.cash=None
        self.action_space=np.arange(3**self.n_stock)

        self.action_list=list(map(list,itertools.product([0,1,2],repeat=self.n_stock))) # Returns sparse matrix of possibilities

        self.state_dim=self.n_stock*2+1
        self.reset()
    
    def reset(self):
        self.cur_step=0
        self.stock_owned=np.zeros(self.n_stock)
        self.stock_price=self.stock_price_history[self.cur_step]
        self.cash=self.initial_investment
        return self._get_obs()

    def step(self,action):
        assert action in self.action_space # If action exists in action space

        prev_val=self._get_val()

        self.cur_step+=1
        self.stock_price=self.stock_price_history[self.cur_step]

        self._trade(action)

        cur_val=self._get_val()

        reward=cur_val=prev_val

        done=self.cur_step==self.n_step-1

        info={"cur_val":cur_val}

        return self._get_obs(),reward,done,info

    def _get_obs(self):
        obs=np.empty(self.state_dim)
        obs[:self.n_stock]=self.stock_owned
        obs[self.n_stock:2*self.n_stock]=self.stock_price
        obs[-1]=self.cash
        return obs
    
    def _get_val(self):
        return self.stock_owned.dot(self.stock_price) + self.cash
    
    def _trade(self, action):
        # index the action we want to perform
        # 0 = sell
        # 1 = hold
        # 2 = buy
        # e.g. [2,1,0] means:
        # buy first stock
        # hold second stock
        # sell third stock
        action_vec = self.action_list[action]

        # determine which stocks to buy or sell
        sell_index = [] # stores index of stocks we want to sell
        buy_index = [] # stores index of stocks we want to buy
        for i, a in enumerate(action_vec):
            if a == 0:
                sell_index.append(i)
            elif a == 2:
                buy_index.append(i)

        # sell any stocks we want to sell
        # then buy any stocks we want to buy
        if sell_index:
            # NOTE: to simplify the problem, when we sell, we will sell ALL shares of that stock
            for i in sell_index:
                self.cash += self.stock_price[i] * self.stock_owned[i]
                self.stock_owned[i] = 0
        if buy_index:
        # NOTE: when buying, we will loop through each stock we want to buy,
        #       and buy one share at a time until we run out of cash
            can_buy = True
            while can_buy:
                for i in buy_index:
                    if self.cash > self.stock_price[i]:
                        self.stock_owned[i] += 1 # buy one share
                        self.cash -= self.stock_price[i]
                    else:
                        can_buy = False



class DQNAgent(object):
    def __init__(self,state_size,action_size):
        self.state_size=state_size
        self.action_size=action_size
        self.memory=ReplayBuffer(state_size,action_size,size=500)
        self.gamma=0.95 # Discount rate
        self.epsilon=1 #Exploration factor
        self.epsilon_min=0.01
        self.epsilon_decay=0.995
        self.model=mlp(state_size,action_size)
    
    def update_replay_memory(self,state,action,reward,next_state,done):
        self.memory.store(state,action,reward,next_state,done)

    def act(self,state):
        if np.random.rand() <=self.epsilon:
            return np.random.choice(self.action_size)
        act_values=self.model.predict(state)
        return np.argmax(act_values[0])

    
    def replay(self,batch_size=32):
        if self.memory.size< batch_size:
            return
        
        minibatch=self.memory.sample_batch(batch_size)
        states=minibatch['s']
        actions=minibatch['a']
        rewards=minibatch['r']
        next_states=minibatch['s2']
        done=minibatch['d']

        target=rewards+self.gamma*np.amax(self.model.predict(next_states),axis=1)

        target[done]=rewards[done]
        ''' For keras we must have the same shape as the prediction. However we only need to update the network for the 
        actions which were actually taken. We can accomplish this by setting the target to be equal to the prediction for all values.
        then only change the targets for the action taken'''
        target_full=self.model.predict(states)
        target_full[np.arange(batch_size),actions]=target

        self.model.train_on_batch(states,target_full)

        if self.epsilon>self.epsilon_min:
            self.epsilon*=self.epsilon_decay
        
    def load(self,name):
        self.model.load_weigths(name)

    def save(self,name):
        self.model.save_weights(name)

def play_one_episode(agent,env,is_train):
    state=env.reset()
    state=scaler.transform([state])
    done=False
    
    while not done:
        action=agent.act(state)
        next_state,reward,done,info=env.step(action)
        next_state=scaler.transform([next_state])
        if is_train == 'train':
            agent.update_replay_memory(state,action,reward,next_state,done)
            agent.replay(batch_size)
        state=next_state

    return info['cur_val']


if __name__=="__main__":
    models_folder='rl_trader_model'
    reward_folder="rl_trader_rewards"
    num_episodes=2000
    batch_size=32
    initial_cash=20000

    parser=argparse.ArgumentParser()
    parser.add_argument('-m','--mode',type=str,required=True,
                        help='either "train" or "test"')
    args=parser.parse_args()

    make_dir(models_folder)
    make_dir(reward_folder)

    data=get_data()
    n_timesteps,n_stocks=data.shape
    n_train=n_timesteps//2

    train_data=data[:n_train]
    test_data=data[n_train:]

    env=MultiStockEnv(train_data,initial_cash)
    state_size=env.state_dim
    action_size=len(env.action_space)
    agent=DQNAgent(state_size,action_size)
    scaler=get_scaler(env)

    pf_value=[]

    if args.mode=='test':
        with open(f'{models_folder}/scaler.pkl','rb') as f:
            scaler = pickle.load(f)
        env=MultiStockEnv(test_data,initial_cash)
        agent.epsilon=0.01
        agent.load(f'{models_folder}/dqn.h5')

    for e in range(num_episodes):
        t0=datetime.now()
        val=play_one_episode(agent,env,args.mode)
        dt=datetime.now()-t0
        print(f'episode: {e+1}/{num_episodes},episode end value {val:.2f},duration {dt}')
        pf_value.append(val)

    if args.mode=='train':
        agent.save(f"{models_folder}/dqn.h5")
        with open(f"{models_folder}/scaler.pkl",'wb') as f:
            pickle.dump(scaler,f)
    
    np.save(f'{reward_folder}/{args.mode}.npy',pf_value)

