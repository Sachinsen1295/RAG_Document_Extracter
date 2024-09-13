import setuptools

_version_ = '0.0.1'
REPO_NAME = 'BOT' 
AUTHOR_NAME = 'Sachinsen1295'
SOURCE_REPO = "BOT"
AUTHOR_EMAIL = "sachin.sen1295@gmail.com"


with open("README.md", "r", encoding="utf-8") as f:
    LONG_DESCRIPTION=f.read()

with open("LICENSE", 'r') as L:
    LICENSE = L.read()

setuptools.setup(
                 name=SOURCE_REPO,
                 version=_version_,
                 author=AUTHOR_NAME,
                 author_email=AUTHOR_EMAIL,
                 description="This is my Deed extraction Bot",
                 long_description=LONG_DESCRIPTION,
                 long_description_content = "text/markdown",
                 url=f"https://github.com/{AUTHOR_NAME}/{REPO_NAME}",
                 license = LICENSE,
                 
                  project_urls={
                      
                    "Bug Tracker": f"https://github.com/{AUTHOR_NAME}/{REPO_NAME}/issues",  
                      
                     },
                      package_dir={"":"src"},
                      packages=setuptools.find_packages(where="src")
                 )