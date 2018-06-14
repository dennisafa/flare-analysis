def test():
    list = [1,2,3,4,5,6,5,4,3,2,1,3,4,5,6]
    listFlare = []
    j = 0
    i = 0
    tempVar = 0
    print (len(list))
    while j < len(list):
        if (list[j] - list[j+1]) < 0:
            while j < len(list) - 2 and list[j] < list[j+1] :
                tempVar =  list[j+1]
                #print(tempVar)
                j+=1
            else:
                if j == len(list) - 2:
                    break
                print(tempVar)
        else:
            while j < len(list) - 2 and list[j] > list[j + 1]:
                tempVar = list[j+1]
                j+=1
            else:
                if j == len(list) - 2:
                    break
                print(tempVar)





test()