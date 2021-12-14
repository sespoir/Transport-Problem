#运输问题求解：使用Vogel逼近法寻找初始基本可行解
import numpy as np
import pandas as pd
import copy

#定义函数TP_vogel，用来实现Vogel法寻找初始基可行解
def TP_vogel(c,a,b):#函数含有3个变量，分别是成本系数矩阵、供给方（产地）的供给向量、需求方（销地）的需求向量
    cost=copy.deepcopy(c)#先将成本矩阵拷贝到cost矩阵中，以便随后对c矩阵进行操作
    x=np.zeros(c.shape)#初始化解变量矩阵
    M=pow(10,9)#运输问题通常是极小化问题，这里定义一个足够大的数M，后期会用到
    for factor in c.reshape(1,-1)[0]:#遍历c中的元素
        while int(factor)!=M:#在c中有元素不等于M时进行循环
            if np.all(c==M):
                break #若c中所有元素均等于M时，跳出整个循环（说明此时任何解变量的添加都会使总成本大幅上升，不满足成本极小化目标）
            else:
                print('c:\n',c)
                #计算罚数
                #获取c的行最小元素向量
                row_mini1=[]
                row_mini2=[]
                for row in range(c.shape[0]):
                    Row=list(c[row,:])
                    row_min=min(Row)
                    row_mini1.append(row_min)#找最小元素，写入row_mini1
                    Row.remove(row_min)
                    row_2nd_min=min(Row)
                    row_mini2.append(row_2nd_min)#找次小元素，写入row_mini2
                #print(row_mini1,'\n',row_mini2)
                r_pun=[row_mini2[i]-row_mini1[i] for i in range(len(row_mini1))]#行罚数=该行次小元素-最小元素
                print('行罚数：',r_pun)
                #获取c的列最小元素向量，过程同上
                col_mini1=[]
                col_mini2=[]
                for col in range(c.shape[1]):
                    Col=list(c[:,col])
                    col_min=min(Col)
                    col_mini1.append(col_min)
                    Col.remove(col_min)
                    col_2nd_min=min(Col)
                    col_mini2.append(col_2nd_min)
                c_pun=[col_mini2[i]-col_mini1[i] for i in range(len(col_mini1))]#列罚数=该列次小元素-最小元素
                print('列罚数：',c_pun)
                pun=copy.deepcopy(r_pun)
                pun.extend(c_pun)
                print('罚数向量：',pun)#将行/列罚数向量连接起来，成为罚数向量，其中前len(r_pun)个元素是行罚数，剩余的是列罚数
                max_pun=max(pun)#最大罚数
                max_pun_index=pun.index(max(pun))#最大罚数索引
                max_pun_num=max_pun_index+1#最大罚数的序数（索引+1）
                print('最大罚数：',max_pun,'元素序号：',max_pun_num)
                if max_pun_num<=len(r_pun):#如果最大罚数的序数<=行罚数的个数，说明最大罚数存在于某行的罚数之中
                    row_num=max_pun_num
                    print('对第',row_num,'行进行操作：')#找到罚数最大的那一行
                    row_index=row_num-1
                    catch_row=c[row_index,:]
                    print(catch_row)
                    min_cost_colindex=int(np.argwhere(catch_row==min(catch_row)))#获取这一行中，最小成本系数的所在列的索引
                    print('最小成本所在列索引：',min_cost_colindex)#由此行此列定位需要填入数值的x变量矩阵位置
                    if a[row_index]<=b[min_cost_colindex]:#填入的变量数值为此行对应的资源向量a和此列对应的b中的较小值
                        x[row_index,min_cost_colindex]=a[row_index]
                        c1=copy.deepcopy(c)
                        c1[row_index,:]=[M]*c1.shape[1]
                        b[min_cost_colindex]-=a[row_index]
                        a[row_index]-=a[row_index]
                        #填入变量后，将已满足条件（资源耗尽/需求满足）的行/列的元素改为M，后期不会再在该行/列中添加变量
                        #同时修改向量a和b
                    else:
                        x[row_index,min_cost_colindex]=b[min_cost_colindex]
                        c1=copy.deepcopy(c)
                        c1[:,min_cost_colindex]=[M]*c1.shape[0]
                        a[row_index]-=b[min_cost_colindex]
                        b[min_cost_colindex]-=b[min_cost_colindex]
                else:#如果最大罚数的序数>行罚数的个数，说明最大罚数存在于某列的罚数之中
                    col_num=max_pun_num-len(r_pun)#该列的序数为最大罚数的序数减去行罚数的个数
                    col_index=col_num-1
                    print('对第',col_num,'列进行操作：')
                    catch_col=c[:,col_index]
                    print(catch_col)
                    #寻找最大罚数所在行/列的最小成本系数
                    min_cost_rowindex=int(np.argwhere(catch_col==min(catch_col)))
                    print('最小成本所在行索引：',min_cost_rowindex)
                    #计算将该位置应填入x矩阵的数值（a,b中较小值）
                    if a[min_cost_rowindex]<=b[col_index]:
                        x[min_cost_rowindex,col_index]=a[min_cost_rowindex]
                        c1=copy.deepcopy(c)
                        c1[min_cost_rowindex,:]=[M]*c1.shape[1]
                        b[col_index]-=a[min_cost_rowindex]
                        a[min_cost_rowindex]-=a[min_cost_rowindex]
                    else:
                        x[min_cost_rowindex,col_index]=b[col_index]
                        #填入后删除已满足/耗尽资源系数的行/列，得到剩余的成本矩阵，并改写资源系数
                        c1=copy.deepcopy(c)
                        c1[:,col_index]=[M]*c1.shape[0]
                        a[min_cost_rowindex]-=b[col_index]
                        b[col_index]-=b[col_index]
                c=c1#将c1传给c，准备进行下一次迭代
                print('本次迭代后的x矩阵：\n',x)
                print('本次迭代后的a矩阵:',a)
                print('本次迭代后的b矩阵:',b)
                print('本次迭代后的c矩阵:\n',c)
            if np.all(c==M):
                print('【迭代完成】')
                print('------------------------------------------------')
            else:
                print('【迭代未完成】')
                print('------------------------------------------------')
    total_cost=np.sum(np.multiply(x,cost))
    if np.all(a==0):
        if np.all(b==0):
            print('>>>供求平衡<<<')
        else:
            print('>>>供不应求，需求方有余量<<<')
    elif np.all(b==0):
        print('>>>供大于求，供给方有余量<<<')
    else:
        print('>>>问题未达到最优<<<')
    #由a和b向量分别的元素和是否相等即可判断供需关系
    print('>>>初始基本可行解x*：\n',x)
    print('>>>当前总重量：',total_cost)

def TP_vogel_matrix(mat):#本函数可以对由c,a,b构成的运输表mat直接求初始基可行解
    c=mat[:-1,:-1]
    a=mat[:-1,-1]
    b=mat[-1,:-1]
    TP_vogel(c,a,b)
#示例
#运输表矩阵为：

#保存为EXCEL表格
path=r'C:\Python\pythonProject8\TP_sample_data.xlsx'#作为输入数据
mat=pd.read_excel(path,header=None).values
TP_vogel_matrix(mat)