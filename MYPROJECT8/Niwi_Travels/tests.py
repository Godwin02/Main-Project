
# Login Functionality Testing
# from django.test import TestCase
# from selenium import webdriver
# from selenium.webdriver.common.by import By
# from selenium.webdriver.support.ui import WebDriverWait
# from selenium.webdriver.support import expected_conditions as EC

# class LoginTestCase(TestCase):
#     def setUp(self):
#         self.driver = webdriver.Chrome()  # Initialize Chrome WebDriver
#         self.driver.implicitly_wait(10)   # Implicit wait for elements

#     def tearDown(self):
#         self.driver.quit()  # Close the browser session after each test

#     def test_login(self):
#         self.driver.get('http://127.0.0.1:8000/log/')

#         username_input = self.driver.find_element(By.NAME, 'username')
#         password_input = self.driver.find_element(By.NAME, 'password')
#         login_button = self.driver.find_element(By.ID, 'login-link')

#         username_input.send_keys('appu')
#         password_input.send_keys('Appu@123')
#         login_button.click()

#         # Wait for the page to load after login (replace this with appropriate page load condition)
#         WebDriverWait(self.driver, 10).until(EC.url_to_be('http://127.0.0.1:8000/thome/'))

#         Perform assertions to verify successful login
#         expected_content = 'Text or Element on Successful Login Page'
#         assert expected_content in self.driver.page_source, f"Expected content '{expected_content}' not found after login"



#Booking Functionality Testing

# from django.test import TestCase
# from selenium import webdriver
# from selenium.webdriver.common.by import By
# from selenium.webdriver.support.ui import WebDriverWait
# from selenium.webdriver.support import expected_conditions as EC
# import os
# class LoginTestCase(TestCase):
#     def setUp(self):
#         self.driver = webdriver.Chrome()
#         self.driver.implicitly_wait(10)

#     def tearDown(self):
#         self.driver.quit()

#     def test_login_and_navigation(self):
#         # Step 1: Login
#         self.driver.get('http://127.0.0.1:8000/log/')

#         username_input = self.driver.find_element(By.NAME, 'username')
#         password_input = self.driver.find_element(By.NAME, 'password')
#         login_button = self.driver.find_element(By.ID, 'login-link')

#         username_input.send_keys('appu')
#         password_input.send_keys('Appu@123')
#         login_button.click()

#         WebDriverWait(self.driver, 10).until(EC.url_to_be('http://127.0.0.1:8000/thome/'))

#         # Step 2: Navigate to the desired page
#         package_id = 5986
#         self.driver.get(f'http://127.0.0.1:8000/package/{package_id}/')

       

#         WebDriverWait(self.driver, 10).until(EC.presence_of_element_located((By.XPATH, '//button[@type="submit"]')))

#         # Step 4: Click the "Book Now" button on the new page
#         book_now_button = self.driver.find_element(By.XPATH, '//button[@type="submit"]')
#         book_now_button.click()

#         WebDriverWait(self.driver, 10).until(EC.url_to_be(f'http://127.0.0.1:8000/add_passenger/{package_id}/'))

#         # Fill in the booking details
#         passenger_limit_input = self.driver.find_element(By.ID, 'passenger-limit')
#         passenger_limit_input.clear()
#         passenger_limit_input.send_keys('2')  # Adjust the number of passengers as needed
#         children_input = self.driver.find_element(By.ID, 'children')
#         children_input.clear()
#         children_input.send_keys('1')  # Adjust the number of children as needed
#         passenger_name_input = self.driver.find_element(By.NAME, 'passenger_name')
#         passenger_name_input.clear()
#         passenger_name_input.send_keys('Passenger')
#         passenger_age_input = self.driver.find_element(By.NAME, 'passenger_age')
#         passenger_age_input.clear()
#         passenger_age_input.send_keys('25')
#         proof_of_id_input = self.driver.find_element(By.NAME, 'proof_of_id')
#         image_filename = 'D:/User/OneDrive/Pictures/Screenshots/Screenshot (2).png'
#         image_path = os.path.abspath(image_filename)
#         proof_of_id_input.send_keys(image_path)
#         # Add more passengers if needed
#         # Submit the form
#         submit_button = self.driver.find_element(By.XPATH, '//button[@type="submit"]')
#         submit_button.click()
#         #You can add assertions here to check if the booking was successful






#Booking Cancellation Testing
# from django.test import TestCase
# from selenium import webdriver
# from selenium.webdriver.common.by import By
# from selenium.webdriver.support.ui import WebDriverWait
# from selenium.webdriver.support import expected_conditions as EC
# from selenium.webdriver.common.action_chains import ActionChains

# import os
# class LoginTestCase(TestCase):
#     # ... (your existing code)
#     def setUp(self):
#         self.driver = webdriver.Chrome()
#         self.driver.implicitly_wait(10)

#     def tearDown(self):
#         self.driver.quit()

#     def test_cancel_booking(self):
#         # Step 1: Login
#         self.driver.get('http://127.0.0.1:8000/log/')
#         # Perform login steps here (as shown in your existing test_login_and_navigation function)
#         self.driver.get('http://127.0.0.1:8000/log/')

#         username_input = self.driver.find_element(By.NAME, 'username')
#         password_input = self.driver.find_element(By.NAME, 'password')
#         login_button = self.driver.find_element(By.ID, 'login-link')

#         username_input.send_keys('appu')
#         password_input.send_keys('Appu@123')
#         login_button.click()

#         WebDriverWait(self.driver, 20).until(EC.url_to_be('http://127.0.0.1:8000/thome/'))

#         # Step 2: Click on the profile dropdown
#         profile_dropdown = self.driver.find_element(By.CLASS_NAME, 'nav-item.dropdown')
#         ActionChains(self.driver).move_to_element(profile_dropdown).click().perform()
#                         # Step 2: Navigate to "Upcoming Journeys"
#         upcoming_journeys_link = self.driver.find_element(By.LINK_TEXT, 'Upcoming Journeys')
#         upcoming_journeys_link.click()

#        # Step 4: Cancel Booking for a specific package
#         # Locate the card based on the package name
#         package_name_to_cancel = "Thekady Stay"
#         card_xpath = f'//h5[text()="{package_name_to_cancel}"]/ancestor::div[@class="card"]'
#         card = self.driver.find_element(By.XPATH, card_xpath)

#         # Find the cancel button within the located card
#         cancel_button_xpath = './/form/button[@name="cancel"]'
#         cancel_button = card.find_element(By.XPATH, cancel_button_xpath)
#         cancel_button.click()
#         # After cancellation, you are redirected to "thome"

#         # Example assertion: Check if the "Make Payment" button is not present after cancellation
#         # with self.assertRaises(NoSuchElementException):
#         #     self.driver.find_element(By.XPATH, cancel_button_xpath + '/../following-sibling::form/button[@class="btn btn-success"]')



from django.test import TestCase
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains

import os
class LoginTestCase(TestCase):
    # ... (your existing code)
    def setUp(self):
        self.driver = webdriver.Chrome()
        self.driver.implicitly_wait(10)

    def tearDown(self):
        self.driver.quit()

    def test_cancel_booking(self):
        # Step 1: Login
        self.driver.get('http://127.0.0.1:8000/log/')
        # Perform login steps here (as shown in your existing test_login_and_navigation function)
        self.driver.get('http://127.0.0.1:8000/log/')

        username_input = self.driver.find_element(By.NAME, 'username')
        password_input = self.driver.find_element(By.NAME, 'password')
        login_button = self.driver.find_element(By.ID, 'login-link')

        username_input.send_keys('appu')
        password_input.send_keys('Appu@123')
        login_button.click()

        WebDriverWait(self.driver, 20).until(EC.url_to_be('http://127.0.0.1:8000/thome/'))

        # Step 2: Click on the profile dropdown
        profile_dropdown = self.driver.find_element(By.CLASS_NAME, 'nav-item.dropdown')
        ActionChains(self.driver).move_to_element(profile_dropdown).click().perform()
                        # Step 2: Navigate to "Upcoming Journeys"
        upcoming_journeys_link = self.driver.find_element(By.LINK_TEXT, 'Upcoming Journeys')
        upcoming_journeys_link.click()

       # Step 4: Cancel Booking for a specific package
        # Locate the card based on the package name
        package_name_to_view = "Thekady Stay"  # Replace with the actual package name
        view_passengers_link_xpath = f'//h5[text()="{package_name_to_view}"]/ancestor::div[@class="card-body"]/p[@class="card-text"]/a[@class="view-details-link"]'
        view_passengers_link = self.driver.find_element(By.XPATH, view_passengers_link_xpath)
        view_passengers_link.click()

        passenger_name_to_delete = "Passenger"  # Replace with the actual passenger name
        delete_passenger_link_xpath = f'//td[text()="{passenger_name_to_delete}"]/following-sibling::td/a[contains(@class, "btn-danger")]'
        delete_passenger_link = self.driver.find_element(By.XPATH, delete_passenger_link_xpath)
        delete_passenger_link.click()
