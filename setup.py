from setuptools import setup
from glob import glob
package_name = "festival_yolo"

setup(
    name=package_name,
    version="0.0.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name, ["launch/yolo.launch.py"]),
        ("share/" + package_name + "checkpoints", glob("checkpoints/*.*")),
        ("share/" + package_name + "/configs", glob("configs/*.*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="jeonghan",
    maintainer_email="kimjh9813@naver.com",
    description="festival ur yolo",
    license="TODO: License declaration",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "yolo = festival_yolo.main:main"
        ],
    },
)