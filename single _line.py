

while True:
    user_inp = int(input("Enter any number"))
    if user_inp == 8:
        print("Loop started again")
        break
    for i in range(user_inp):
        print(i)
    