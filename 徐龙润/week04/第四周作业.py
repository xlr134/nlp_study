Dict = {
    "经常":0.1,
    "经":0.05,
    "有":0.1,
    "常":0.001,
    "有意见":0.1,
    "歧":0.001,
    "意见":0.2,
    "分歧":0.2,
    "见":0.05,
    "意":0.05,
    "见分歧":0.05,
    "分":0.1
}
sentence = "经常有意见分歧"
target = []
def dfs(sentence,s_index,t,index):
    if s_index==len(sentence):
        if t[index] in Dict.keys():
           target.append(t)
        return None
    else:
        xuan_t = t.copy()
        xuan_t[index] += sentence[s_index]
        dfs(sentence,s_index+1,xuan_t.copy(),index)
        if t[index] in Dict.keys():
            temp = t.copy()
            temp.append(sentence[s_index])
            t_index = index+1
            dfs(sentence,s_index+1,temp,t_index)
def all_cut():
    t = [sentence[0]]
    dfs(sentence,1,t,0)

all_cut()
print(f"target = {target}")
