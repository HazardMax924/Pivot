import numpy as np
import random
import operator as op
import time

path='C:\\Users\\minha\\Desktop\\数据集'
dimension=15             #数据特征长度
m=100                    #mBF的长度
k=30                     #hash函数数量
t=0.5                    #相似阈值
# ratio=np.array([34,34,34,34,34,33,33,33,33,33,33,33,33,33,33])
ratio=np.array([7,7,7,7,7,7,7,7,7,7,6,6,6,6,6,6])
# size=['1000','2000''3000','4000''5000','6000''7000''8000''9000''10000']
# match_rate=['0.1','0.2''0.3''0.4''0.5','0.6''0.7''0.8''0.9''1.0']
size=['1000','2000''3000','4000''5000','6000''7000''8000''9000''10000']
match_rate=['0.1','0.2''0.3''0.4''0.5','0.6''0.7''0.8''0.9''1.0']

#BKDRHash
def hashs(p,hash_num,bf_size):
    """
    进行hash计算
    p:插入的字符串
    hash_num:第几个hash函数
    bf_size:bf的大小
    return:返回的哈希值
    """
    seeds=np.arange(2001,4001,2)
    hash=0;
    for i in range(0,len(p)):
        hash=(hash*seeds[hash_num]+ord(p[i]))%bf_size
    return hash;

def insert(p,hash_need_num,bf):
    """
    进行hash插入
    p:插入的字符串
    hash_need_num:一共要执行的hash数量
    bf:返回的bf
    return:bf
    """
    n=len(bf)
    for i in range(0,hash_need_num):
        bf[hashs(p,i,n)]=1
    return bf

def lookup(p,hash_need_num,bf):
    """
    进行bf查找
    p:插入的字符串
    hash_need_num:一共要执行的hash数量
    bf:在哪个bf里查找
    return:flag 0查找失败1成功
    """
    flag=1 
    n=len(bf)
    for i in range(0,hash_need_num):
        if bf[hashs(p,i,n)]==0:
            flag=0
            break
    return flag

def dice(a,b):
    under=sum(a)+sum(b);
    together=np.sum(a*b)*2
    ans=together/under
    return ans

def hamming_dis(a,b):
    under=sum(a)+sum(b);
    together=np.sum(a*b)
    ans=under-together*2
    return ans

def rad(a):
    return sum(a)*(1-t)/t

def FFT_get_pivot(a,num):
    #a 所需确定的数组，num数组的长度
    distance=np.zeros(num)  #0未被访问，-1已经访问过
    flag=np.zeros(num);      #0未被访问，1已经访问过
    sum_flag=num
    i=0
    while sum_flag!=1:
        distance[i]=-1
        flag[i]=1
        sum_flag-=1
        maxi=-1;
        key=-1;
        for j in range(0,num):
            if flag[j]==1:
                continue
            distance[j]=hamming_dis(a[i],a[j])
            if (distance[j]>maxi):
                maxi=distance[j]
                key=j
        i=key
    return i
    
def get_radius(array_a,num):
    #array_a 传入的数组，num数组长度
    a=np.zeros(num)
    for i in range(0,num):
        a[i]=rad(array_a[i])
    return a

def caculate(a,b,c,d,e,f):
    #a表示正确的链接，b表示错误的链接,c为in1，b为in2
    correct=a
    wrong=b
    fn_p=1 
    tn_p=1
    for i in range(0,len(wrong),2):
        if ((wrong[i]==wrong[i+1]) and (wrong[i]<=int(c)*float(d))) or (dice(bf[wrong[i]],bf[wrong[i+1]])>t):
            fn_p+=1
        else:
            tn_p+=1
    fp_p=1
    tp_p=1
    for i in range(0,len(correct),2):
        if (correct[i]==correct[i+1]) and (correct[i]<=int(c)*float(d)) and (dice(bf[correct[i]],bf[correct[i+1]])>t):
            tp_p+=1
        else:
            fp_p+=1
    recall=tp_p/(tp_p+fn_p)
    precision=tp_p/(tp_p+fp_p)
    f1=2*recall*precision/(precision+recall)
    print("tp fp tn fn:")
    print(tp_p,fp_p,tn_p,fn_p)
    return recall,precision,f1


for in1 in size:
    for in2 in match_rate:
        inputname1=path+'\\'+in1+'\\'+in2+'\\相同数据 - 副本.txt'
        inputname2=path+'\\'+in1+'\\'+in2+'\\相同数据 - 副本 (2).txt'
        inputname3=path+'\\'+in1+'\\'+in2+'\\相同数据 - 副本 (3).txt'
        num=int(in1)                #数据数量

        data=np.zeros(shape=(num,dimension)).astype(np.str_)
        data[data == '0.0'] = ''
        cnt=0;
        #对文件1处理
        with open(inputname1, 'r', encoding='utf-8') as f:
            for ann in f.readlines():
                ann = ann.strip('\n').replace('"','') .split('\t')       #去除文本中的换行符与\t
                data[cnt]=ann;
                cnt+=1
                if cnt==num:
                    break

        #对文件2处理
        data2=np.zeros(shape=(num,dimension)).astype(np.str_)
        data2[data2 == '0.0'] = ''
        cnt=0;
        with open(inputname2, 'r', encoding='utf-8') as f2:
            for ann2 in f2.readlines():
                ann2 = ann2.strip('\n').replace('"','') .split('\t')       #去除文本中的换行符与\t
                data2[cnt]=ann2;
                cnt+=1
                if cnt==num:
                    break
        
        #对文件3处理
        data3=np.zeros(shape=(num,dimension)).astype(np.str_)
        data3[data3 == '0.0'] = ''
        cnt=0;
        with open(inputname3, 'r', encoding='utf-8') as f3:
            for ann3 in f3.readlines():
                ann3 = ann3.strip('\n').replace('"','') .split('\t')       #去除文本中的换行符与\t
                data3[cnt]=ann3;
                cnt+=1
                if cnt==num:
                    break

        bf=np.zeros((num,m))
        for i in range(0,num):
            mbf=np.zeros((dimension,m))
            for j in range(0,dimension):
                if len(data[i][j])==0:
                    continue
                temp=' '+data[i][j][0]
                mbf[j]=insert(temp,k,mbf[j])
                for k in range(0,len(data[i][j])-2):
                    mbf[j]=insert(data[i][j][k:k+1],k,mbf[j])
                temp=data[i][j][len(data[i][j])-1]+' ';
                mbf[j]=insert(temp,k,mbf[j])

        bf2=np.zeros((num,m))
        for i in range(0,num):
            mbf2=np.zeros((dimension,m))
            for j in range(0,dimension):
                if len(data2[i][j])==0:
                    continue
                temp2=' '+data2[i][j][0]
                mbf2[j]=insert(temp2,k,mbf2[j])
                for k in range(0,len(data2[i][j])-2):
                    mbf2[j]=insert(data2[i][j][k:k+1],k,mbf2[j])
                temp2=data2[i][j][len(data2[i][j])-1]+' ';
                mbf2[j]=insert(temp2,k,mbf2[j])

        bf3=np.zeros((num,m))
        for i in range(0,num):
            mbf3=np.zeros((dimension,m))
            for j in range(0,dimension):
                if len(data3[i][j])==0:
                    continue
                temp3=' '+data3[i][j][0]
                mbf3[j]=insert(temp3,k,mbf3[j])
                for k in range(0,len(data3[i][j])-2):
                    mbf3[j]=insert(data3[i][j][k:k+1],k,mbf3[j])
                temp3=data3[i][j][len(data3[i][j])-1]+' ';
                mbf3[j]=insert(temp3,k,mbf3[j])

        #生成rbf
        for i in range(0,num):
            slice=[]
            slice2=[]
            slice3=[]
            for j in range(0,dimension):
                for k in range(0,ratio[j]):
                    state = np.random.randint(0,m)
                    slice.append(mbf[j][state])
                    slice2.append(mbf2[j][state])
                    slice3.append(mbf3[j][state])
            bf[i]=np.array(slice, dtype = int)
            bf2[i]=np.array(slice2, dtype = int)
            bf3[i]=np.array(slice2, dtype = int)

            state = np.random.get_state()
            np.random.shuffle(bf[i])
            np.random.set_state(state)
            np.random.shuffle(bf2[i])
            np.random.set_state(state)
            np.random.shuffle(bf3[i])

        print("RBF生成完毕")
        start=time.time()
        #枢轴选取部分采用FFT方法
        pivot_index=FFT_get_pivot(bf,num)
        rad_p=get_radius(bf,num)
        r=rad(bf[pivot_index])
        dis_indexes=[]  #枢轴所存的记录距离
        i_indexes=[];
        for i in range(num):
            temp=hamming_dis(bf[pivot_index],bf[i])
            if temp<r:
                dis_indexes.append(temp)
                i_indexes.append(i);

        wrong=[]
        correct=[]

        for i in range(0,num):
            if (i%1000==0):
                print("枢轴计算次数1:",i)
            rbf2=rad(bf2[i])
            dpq=hamming_dis(bf[pivot_index],bf2[i])
            if dpq>rad_p[pivot_index]+rbf2:
                break;
            else:
                for k in range(0,len(i_indexes)):
                    if dpq-dis_indexes[k]>rbf2:
                        wrong.append(i)
                        wrong.append(k)
                        # print(i,i_indexes[j][k],'wrong')
                    else:
                        correct.append(i)
                        correct.append(k)
                        # print(i,i_indexes[j][k],'correct')
        end=time.time()
        print("枢轴时间1:%.2f"%(end-start))

        pivot_index=FFT_get_pivot(bf3,num)
        r=rad(bf3[pivot_index])
        n=len(correct)
        i=0
        correct2=correct
        while i<n-2:
            if (hamming_dis(bf3[pivot_index],bf[correct[i]])>r) or (hamming_dis(bf3[pivot_index],bf[correct[i+1]])>r):
                wrong.append(correct[i])
                wrong.append(correct[i+1])
            else:
                correct2.append(correct[i])
                correct2.append(correct[i+1])
            i+=2
        correct=correct2

        end=time.time()
        print("枢轴时间2:%.2f"%(end-start))
        #统计枢轴fn、fp、tn、tp
        recall,precision,f1=caculate(correct,wrong,in1,in2,bf,bf2)
        print("枢轴recall、precision与f1:")
        print(recall,precision,f1)
        end=time.time()
        print("枢轴总时间:%.2f"%(end-start))

        wrong_bf=[]
        correct_bf=[]
        start=time.time()
        for i in range(0,num):
            if (i%1000==0):
                print("无枢轴计算次数：",i)
            for j in range(0,num):
                ans=dice(bf[i],bf2[j])
                if ans>t:
                    for k in range(0,num):
                        ans=dice(bf2[j],bf3[k])
                        if (ans>t):
                            correct_bf.append(i)
                            correct_bf.append(j)
                            correct_bf.append(k)
                else:
                    wrong_bf.append(i)
                    wrong_bf.append(j)
            # print()
        end=time.time()
        print("无枢轴时间：%.2f"%(end-start))
        # for i in range(0,len(correct_bf)-2):
        #     print(correct_bf[i],correct_bf[i+1],correct_bf[i+2])
        #     i+=3
        # for i in range(0,len(wrong_bf)):
        #     print(wrong_bf[i],wrong_bf[i+1])
        #     i+=2

        #统计无枢轴fn、fp、tn、tp
        fn=1 
        tn=1
        for i in range(0,len(wrong_bf),2):
            if (wrong_bf[i]==wrong_bf[i+1]):
                fn+=1
            else:
                tn+=1
        fp=1 
        tp=1
        for i in range(0,len(correct_bf),3):
            if (correct_bf[i]==correct_bf[i+1]==correct_bf[i+2]) and (correct_bf[i]<=int(size[0])*float(match_rate[0])):
                tp+=1
            else:
                fp+=1

        recall=tp/(tp+fn)
        precision=tp/(tp+fp)
        f1=2*recall*precision/(precision+recall)
        print("无枢轴recall、precision与f1:")
        print(recall,precision,f1)
        print("tp fp tn fn:")
        print(tp,fp,tn,fn)
        print()