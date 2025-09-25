import uuid

random_uuid = uuid.uuid4()
hex_uuid = random_uuid.hex
print(hex_uuid)
#uuid4()是uuid模块中的一个函数，用于生成随机版本的UUID，也被称为版本4的UUID。
# 这种UUID是通过随机数生成的，在理论上具有极低的重复概率。
#hex是UUID对象的一个属性，用于获取UUID的十六进制表示形式，
#它会返回一个去掉连字符的32位十六进制字符串。