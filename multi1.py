from numpy.core.numeric import full
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import seaborn as sns
from sklearn.linear_model import LinearRegression
df = pd.read_csv("Salary_Data.csv")
a = df['YearsExperience']
b = df['Age']
Gender = {'M': 1,'F': 2}
df.Gender = [Gender[item] for item in df.Gender]
c= df['Gender']
d=df['Salary']
def before_multicollinearity():
    a_mean = np.mean(a)
    b_mean = np.mean(b)
    c_mean = np.mean(c)
    num = 0
    den = 0
    for i in range(len(a)):
        num += (a[i] - a_mean) * (b[i] - b_mean)
        den += (a[i] - a_mean) ** 2
    b1 = (num / den)
    b0 = b_mean - (b1 * a_mean)
    print("Slope=", b1)
    print("Intercept=", b0)
    num1 = 0
    den1 = 0
    for i in range(len(b)):
        num1 += (b[i] - b_mean) * (c[i] - c_mean)
        den1 += (b[i] - b_mean) ** 2
    c1 = (num1 / den1)
    c0 = c_mean - (c1 * b_mean)
    print("Slope=", c1)
    print("Intercept=", c0)
    num2 = 0
    den2 = 0
    for i in range(len(c)):
        num2 += (c[i] - c_mean) * (a[i] - a_mean)
        den2 += (c[i] - c_mean) ** 2
    d1 = (num2 / den2)
    d0 = a_mean - (d1 * c_mean)
    print("Slope=", d1)
    print("Intercept=", d0)
    ap = []
    bp = [] 
    cp = []
    for i in range(len(a)):
        bp.append(b1*a[i] + b0)
    for i in range(len(b)):
        cp.append(c1*b[i] + c0)
    for i in range(len(c)):
        ap.append(d1*c[i] + d0)
    sst_1=0
    ssr_1=0
    sse_1=0
    for i in range(len(b)):
        sst_1 += (b[i] - b_mean)**2
    for i in range(len(bp)):
        ssr_1 += (bp[i] - b_mean)**2
        sse_1 += (b_mean - bp[i])**2
    r_sq1 =ssr_1/sst_1
    vif_1 = 1/(1-(r_sq1)**2)
    #print(sst_1,ssr_1,sse_1)
    print("R square value for a and b =", r_sq1)
    print("Vif=", vif_1)
    sst_2=0
    ssr_2=0
    sse_2=0
    for i in range(len(c)):
        sst_2 += (c[i] - c_mean)**2
    for i in range(len(cp)):
        ssr_2 += (cp[i] - c_mean)**2
        sse_2 += (c_mean - cp[i])**2
    r_sq2 =ssr_2/sst_2
    vif_2 = 1/(1-(r_sq2)**2)
    #print(sst_2,ssr_2,sse_2)
    print("R square value for b and c =", r_sq2)
    print("Vif=", vif_2)
    sst_3=0
    ssr_3=0
    sse_3=0
    for i in range(len(a)):
        sst_3 += (a[i] - a_mean)**2
    for i in range(len(ap)):
        ssr_3 += (ap[i] - a_mean)**2
        sse_3 += (a_mean - ap[i])**2
    r_sq3 =ssr_3/sst_3
    vif_3 = 1/(1-(r_sq3)**2)
    #print(sst_3,ssr_3,sse_3)
    print("R square value for c and a =", r_sq3)
    print("Vif=", vif_3)
    plt.rcParams["figure.figsize"] = (12,6)
    plt.subplot(1,3,1)
    plt.scatter(a,b, color='violet' ,label='Actual Values')
    plt.plot(a, bp, color='black', label='Predicted Values')
    plt.xlabel("Years Of Experience")
    plt.ylabel("Age")
    plt.legend()
    plt.subplot(1,3,2)
    plt.scatter(b,c, color='orange' ,label='Actual Values')
    plt.plot(b, cp, color='red', label='Predicted Values')
    plt.xlabel("Age")
    plt.ylabel("Gender")
    plt.legend()
    plt.subplot(1,3,3)
    plt.scatter(c,a, color='lightgreen' ,label='Actual Values')
    plt.plot(c, ap, color='indigo', label='Predicted Values')
    plt.xlabel("Gender")
    plt.ylabel("Years Of Experience")
    plt.suptitle('Multicollinearity', color='black', size=20)
    plt.legend()
    plt.show()

def after_multicollinearity():
    x = df.drop(['Age','Salary'],axis=1)
    print(x)

def multiple_regression():
    df['X1^2']=df['YearsExperience']**(2)
    df['X2^2']=df['Gender']**(2)
    df['X1*Y']=df['YearsExperience']*df['Salary']
    df['X2*Y']=df['Gender']*df['Salary']
    df['X1*X2']= df['YearsExperience']*df['Gender']
    X12s= df['X1^2'].sum()
    X22s= df['X2^2'].sum()
    X1Ys=df['X1*Y'].sum()
    X2Ys=df['X2*Y'].sum()
    X1X2s= df['X1*X2'].sum()
    X1s= df['YearsExperience'].sum()
    X2s=df['Gender'].sum()
    Ys= df['Salary'].sum()
    n=30
    x12s= X12s- (X1s**2)/n
    x22s= X22s- (X2s**2)/n
    x1ys= X1Ys- (X1s*Ys)/n
    x2ys= X2Ys- (X2s*Ys)/n
    x1x2s= X1X2s- (X1s*X2s)/n
    b1= (x22s*x1ys-x1x2s*x2ys)/(x12s*x22s-(x1x2s**2))
    b2= (x12s*x2ys-x1x2s*x1ys)/(x12s*x22s-(x1x2s**2))
    b0= (Ys-b1*X1s-b2*X2s)/n
    print('b0:' ,b0)
    print('b1:', b1)
    print('b2:',b2)
    print(df)
    y_est=b0+ b1*a+b2*c
    fig = plt.figure(figsize = (10, 7))
    ax = plt.axes(projection ="3d")
    ax.scatter3D(a, d, c, color = "green")
    ax.plot3D(a, y_est, c, color = "red")
    plt.xlabel('Years Of Experience')
    plt.ylabel('Salary')
    plt.show()

print("""Press 1 - Before Detecting Multicollinearity 
Press 2- After Detecting Multicollinearity
Press 3 - multiple regression""")
print("Enter Your Choice:")
n1 = int(input())
if n1 == 1:
    before_multicollinearity()
elif n1 == 2:
    after_multicollinearity()
elif n1 == 3:
    multiple_regression()
else:
    print("Invalid Choice Entered")
