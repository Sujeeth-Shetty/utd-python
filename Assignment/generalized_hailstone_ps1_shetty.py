import itertools 

def hailstone(a,b,n):
    steps=0
    temp=n
    pattern=list()
    #loop through untill the original number repeats or untill a holding pattern is recognized with a different stating point
    while (n!=temp) or (n==temp and steps==0) or (find_repeat(pattern)!=True and n!=temp):
        pattern.append(n)
        steps=steps+1

        #If odd
        if n%2!=0:
            n= a*n+b
        else:
        #if even    
            n=n//2
        
        #break infinite loop
        if steps>100:
            return 0,pattern
    
    return steps,pattern

#count only unique holding patterns
def count_unique(p, count):
    for i in range(len(p)):
        for j in range(i+1,len(p)):
            p[j].sort()
            p[i].sort()
            if len(p[j])== len(p[i]) and p[j]==p[i]:
                count=count-1
                break;
    return count

#find holding pattern other than starting number
def find_repeat(numbers):
    seen = set()
    for num in numbers:
        if num in seen:
            return True
        seen.add(num)

#Main Program
count=0
dpattern=list()

#for a=1 to 10 & b=1 to 10
for a,b in itertools.product(range(1,11), range(1,11)):
    #for original number 1 to 1000
    for n in range(1,1000):
        steps,pattern=hailstone(a,b,n)
        if steps !=0:
            dpattern.append(pattern)
            count=count+1

    print('Holding patterns for', a,',',b,'=',count_unique(dpattern,count))
    dpattern.clear()
    count=0
