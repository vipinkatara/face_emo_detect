import requests

# server url
URL = "http://104.211.73.222:8080/predict"

if __name__ == "__main__":
    # open files
    #file = open(FILE_PATH, "rb")
    #print(file)
    url = {'url':'https://image.shutterstock.com/image-photo/portrait-sad-man-260nw-126009806.jpg'}
    # package stuff to send and perform POST request
    values = url
    # print(type(values))
    response = requests.post(URL, json=values)
    data = response.json()

    print("Predicted keyword: {}".format(data["keyword"]))