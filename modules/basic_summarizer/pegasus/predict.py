# -*- coding: utf-8 -*-
"""
Copyright (c) 2022 Martian.AI, All Rights Reserved.

summary predict module

Authors: apollo2mars(apollo2mars@gmail..com)
Date: 2022/11/22 17:23:06
"""
from transformers import PegasusForConditionalGeneration
from tokenizers_pegasus import PegasusTokenizer

model_path = "../../../resources/embedding/IDEA-CCNL/Randeng-Pegasus-522M-Summary-Chinese"
model = PegasusForConditionalGeneration.from_pretrained(model_path)
tokenizer = PegasusTokenizer.from_pretrained(model_path)

text = """推开丁仪那套崭新的三居室的房门，汪淼闻到了一股酒味，看到丁仪躺在沙发上，电视开着，他的双眼却望着天花板。
汪淼四下打量了一下，看到房间还没怎么装修，也没什么家具和陈设，宽大的客厅显得很空，最显眼的是客厅一角摆放的一张台球桌。"""
# text = """在北京冬奥会自由式滑雪女子坡面障碍技巧决赛中，中国选手谷爱凌夺得银牌。祝贺谷爱凌！今天上午，自由式滑雪女子坡面障碍技巧决赛举行。
# 决赛分三轮进行，取选手最佳成绩排名决出奖牌。
# 第一跳，中国选手谷爱凌获得69.90分。在12位选手中排名第三。完成动作后，谷爱凌又扮了个鬼脸，甚是可爱。
# 第二轮中，谷爱凌在道具区第三个障碍处失误，落地时摔倒。获得16.98分。网友：摔倒了也没关系，继续加油！
# 在第二跳失误摔倒的情况下，谷爱凌顶住压力，第三跳稳稳发挥，流畅落地！获得86.23分！
# 此轮比赛，共12位选手参赛，谷爱凌第10位出场。网友：看比赛时我比谷爱凌紧张，加油！"""

text = """
她知道，与上一个时代相比，掩体世界远不是理想社会，向太阳系边 缘的大移民使得早已消失的一些社会形态又出现了，但这不是倒退而是 螺旋形上升，是开拓新疆域必然出现的东西。
从太平洋一号出来后，曹彬还带程心看了几座特异构型的太空城，其 中距太平洋一号较近的是一座轮辐状城市，就是程心六十多年前曾经到 过的地球太空电梯终端站的放大版。程心对太空城未全部建造成轮辐状 一直不太理解，因为从工程学角度来看，轮辐状是太空城最理想的构型， 建造它的技术难度要远低于整体外壳构型的太空城，建成后具有更高的 强度和抗灾能力，而且便于扩建。
“世界感。”曹彬的回答很简单。。 “什么？” 就是身处一个世界的感觉。太空城必须拥有广阔的内部空间。有开阔的视野，人在里面才能感觉到自己是生活在一个世界中。如果换成轮 辐构型，那人们将生活在一圈或者几圈大竹子里，虽然内人面积与整体外 壳构型的太空城差不多，但里面的人总感觉是在飞船上。“
还有一些构型更为奇特的太空城，它们大多是工业或农业城市，没有 常住人口。比如一座叫资源一号空城，长度达到一百二十千米，直径却只有三十千米，是一根细长的杆子，它并不是绕自己的长轴旋转，而是以 中点为轴心翻着筋斗。这座太空城内部是分层的，不同层域的重力差异 极大。只有少数几层适合居住，其余部分都是适合不同重力的工业区。据 曹彬说，在土星和天王星城市群落，两个或几个杆状太空城可以自中部绞 结在一起。形成十字形或星形的组合体。
掩体工程最早建成的太空城群落是木星和土星群落，在较晚建设的天王星和海王星群落中，出现了一些新的太空城建设理念，其中最重要的是城市接口。在这两个处于太阳系遥远边缘的群落中，每座太空城都带有一个或多个标准接口，可以相互对接组合，组合后的城市居民的流动空间成倍扩大，有着更好的世界感，对社会经济的发展具有重大意义。连通后的大气和生态系统成为一个整体，运行状态更为稳定。目前的城市对接方式一般为同轴对接，这样对接后可以同轴旋转，保持对接前的重力环境不变。也有平行对接或垂直对接的设想，这样可以使组合后的城市空间在各个方向更为均衡，而不仅仅是同轴组合的纵向扩展，但由于组合体共同旋转将使原有的重力环境发生重大改变，所以没有进行过实际尝试。
目前，最大的城市组合体在海王星，八座太空城中的四个同轴组合为一体，形成一个长达两百千米的组合城。在需要的时候，比如黑暗森林打击警报出现时，组合体可以在短时间内分解，以增强各自的机动能力。人们 都抱有一个希望——有一天能够使每个城市群落中的所有太空城合为一体，形成四个整体世界。 目前，在木星、土星、天王星和海王星的背阳面，共有六十四座大型太空城，还有近百座中等和小型太空城以及大量空间站，在由它们构成的掩 体世界中，生活着九亿人。
这几乎是现存人类的全部，在黑暗森林打击到来前，地球文明已经进 入掩体。
每座太空城的政治地位相当于一个国家，四个城市群落共同组成太 阳系联邦，原联合国演变成联邦政府。历史上地球各大文明都曾出现过 城邦时代，现在，城邦世界在太阳系的外围再现了。
地球已经成为一个人烟稀少的世界，只有不到五百万人生活在那里， 那是些不愿离开母星家园、对随时可能到来的死神无所畏惧的人。掩体 世界中也有许多胆大的人不断地前往地球旅游或度假，每次行程都是赌 命的冒险之旅。随着时间的推移，黑暗森林打击日益临近，人们也融入了 掩体世界的生活，对母星的怀念在为生计的忙碌中渐渐淡漠。去地球的 人一天比一天少了，公众也不再关注来自母亲行星的信息，只知道大自然 在重新占领那里的一切，各个大陆都逐渐被森林和草原所覆盖。人们也 听说留下的人都过得像国王一样，每个人都住在宽阔的庄园里，都有自己 的森林和湖泊，但出家门必须带枪，以防野兽的袭击。整个地球世界目前 只是太阳系联邦中的一个普通城邦。
程心和曹彬乘坐的太空艇现在已经航行在木星城市群落的最外侧 在巨大阴暗的木星之畔，这个太空城群落显得那么渺小孤单，仿佛是一面 高大山崖下的几幢小屋，它们远远地透出柔和的烛光，虽然微弱，却是这 无边的严寒和荒寂中仅有的温暖栖所，是所有疲惫旅人的向往。这时，程 心的脑海中竟冒出一首中学时代读过的小诗，是中国民国时期一个早被 遗忘的诗人写的：太阳落下去了， 山、树、石、河， 一切伟大的建筑都埋在黑影里； 人类很有趣地点了他们的小灯： 喜悦他们所看见的； 希望找着他们所要的。
"""

print(">>> INPUT TEXT is >>>")
print(text)
print(">>> INPUT TEXT Lenght is >>>")
print(len(text))
inputs = tokenizer(text, max_length=1024, return_tensors="pt")

# Generate Summary
summary_ids = model.generate(inputs["input_ids"], max_length=256)
summary_tokens = tokenizer.batch_decode(summary_ids,
                                        skip_special_tokens=True,
                                        clean_up_tokenization_spaces=False)
print(summary_tokens)