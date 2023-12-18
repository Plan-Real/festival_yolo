from setuptools import setup
from glob import glob
package_name = "snapbot_yolo"
utils = "snapbot_yolo/utils"

setup(
    name=package_name,
    version="0.0.0",
    packages=[package_name,  utils], 
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name, ["launch/yolo.launch.py"]),
        ("share/" + package_name + "/checkpoints", glob("checkpoints/*.*")),
        ("share/" + package_name + "/configs", glob("configs/*.*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="jeonghan",
    maintainer_email="kimjh9813@naver.com",
    description="snapbot ur yolo",
    license="TODO: License declaration",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "yolo = snapbot_yolo.main:main"
        ],
    },
)
