from locust import HttpUser, task, between

class UploadTest(HttpUser):
    wait_time = between(1, 3)

    @task
    def upload_image(self):
        with open("/Users/mac/Documents/MoonArc/Tested_images/nm.jpeg", "rb") as img:
            self.client.post("/predict", files={"file": img})

