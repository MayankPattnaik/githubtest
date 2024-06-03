import math,copy
import numpy as np
import matplotlib.pyplot as plt
#from lab_utils_uni import plt_house_x,plt_contour_wgrad,plt_divergence,plt_gradients

def compute_cost(x,y,w,b):
    m=len(x)
    cost_sum=0
    for i in range(m):
        f_wb=w*x[i]+b
        cost=(f_wb-y[i])**2
        cost_sum+=cost
        total_cost=(1/(2*m))*cost_sum
    return total_cost

def compute_grad(x,y,w,b):
    m=len(x)
    dj_dw=0
    dj_db=0
    for i in range(m):
        f_wb=w*x[i]+b
        dj_dw_i=(f_wb-y[i])*x[i]
        dj_db_i=(f_wb-y[i])
        dj_dw+=dj_dw_i
        dj_db+=dj_db_i
    dj_dw/=m
    dj_db/=m
    return dj_dw,dj_db

def grad_desc(x,y,w_in,b_in,a,num_iters,cost_func,grad_func):
    J_hist=[]
    p_hist=[]
    b=b_in
    w=w_in
    for i in range(num_iters):
        dj_dw,dj_db=grad_func(x,y,w,b)
        b-=a*dj_db
        w-=a*dj_dw
        if i<100000:
            J_hist.append(cost_func(x,y,w,b))
            p_hist.append([w,b])
        if i%math.ceil(num_iters/10)==0:
            print(f"Iteration {i:4}: Cost {J_hist[-1]:0.2e}",
                  f"dj_dw: {dj_dw:0.3e}, dj_db: {dj_db:0.3e}",
                  f"w: {w:0.3e}, b: {b:0.5e}")
    return w,b,J_hist,p_hist


x_train=np.array([1.0,2.0])
y_train=np.array([300.0,500.0])
w_init=0
b_init=0
iterations=10000
tmp_a=1.0e-2
w_final,b_final,J_hist,p_hist=grad_desc(x_train,y_train,w_init,b_init,tmp_a,iterations,compute_cost,compute_grad)
print(f"(w,b) found by grad_desc:({w_final :8.4f},{b_final:8.4f})")
fig, (ax1,ax2)=plt.subplots(1,2,constrained_layout=True,figsize=(12,4))
ax1.plot(J_hist[:100])
ax2.plot(1000+np.arange(len(J_hist[1000:])),J_hist[1000:])
ax1.set_title("Cost vs iteration(start)");ax2.set_title("Cost vs.Iteration(End)")
ax1.set_ylabel('Cost'); ax2.set_ylabel('Cost')
ax1.set_xlabel('Iteration Step');ax2.set_xlabel('Iteration Step')
plt.show()
print(f"1000 sqft predic {w_final*1.0+b_final:0.1f}")
print(f"2000 sqft predic {w_final*2.0+b_final:0.1f}")
print(f"3000 sqft predic {w_final*3.0+b_final:0.1f}")
print(f"3000 sqft predic {w_final*3.0+b_final:0.1f}")
