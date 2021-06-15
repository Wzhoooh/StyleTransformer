import datetime

# we are saving in CSV style
def add_user(user):
    f = open("users.txt", "a")
    f.write(str(datetime.datetime.now()) + "\t" + str(user.id) + "\t" + 
        str(user.mention) + "\t" + str(user.full_name) + "\t" + str(user.locale) + "\n")
    f.close()

