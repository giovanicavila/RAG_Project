def fizzBuzz(upTo: int):
    upToList = []
    for i in range(upTo):
        if i % 3 == 0 and i % 5 == 0:
            upToList.append("FizzBuzz")
        elif i % 3 == 0 and i % 5 != 0:
            upToList.append("Fizz")
        elif i % 3 != 0 and i % 5 == 0:
            upToList.append("Buzz")
        else:   
            upToList.append(i)
    
    return " ".join(map(str, upToList))


print(fizzBuzz(35))
