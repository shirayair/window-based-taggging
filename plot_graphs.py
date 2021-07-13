import matplotlib.pyplot as plt

# y_pos_acc_1 = [86.66, 87.21, 88.03, 88.28, 88.30, 88.35, 88.51, 88.38, 88.36, 88.61]
# y_pos_loss_1 = [2.999,2.993 , 2.985, 2.983,2.982 ,2.982 ,2.980 ,2.982 ,2.982 , 2.979]
#
#
# y_pos_acc_3_no_embed = [88.98, 89.05,89.31 ,89.20 ,89.51 , 89.33,89.49 , 89.27,89.22, 89.21]
# x = range(1, 11)
#
# plt.plot(x, y_pos_acc_3_no_embed, '-')
# plt.xlabel('iterations')
# plt.ylabel('accuracy')
# plt.ylim([88.8, 90])
# plt.title('POS accuracy on the dev set')
# plt.legend()
# plt.savefig("pos_acc_3_no_embed.png")

y_ner_acc_1 = [63.75, 67.25, 68.50, 69.80, 70.87, 70.60, 70.14, 70.98, 71.33, 71.57, 72.16,
               72.15, 73.09, 72.15, 72.90, 72.39, 72.87, 72.50, 73.20, 73.13, 74.26, 73.53, 73.60,
               73.36, 73.81, 73.55, 74.18, 73.58, 73.48, 74.02]
y_ner_loss_1 = [1.204, 1.191, 1.176, 1.183, 1.172, 1.179, 1.173, 1.170, 1.173, 1.175,
                1.168, 1.177, 1.167, 1.176, 1.177, 1.168, 1.175, 1.172, 1.171, 1.172,
                1.172, 1.169, 1.175, 1.173, 1.174, 1.176, 1.172, 1.178, 1.178, 1.177]

x = range(1, 31)

plt.plot(x, y_ner_loss_1, '-')
plt.xlabel('iterations')
plt.ylabel('loss')
plt.ylim([1.14, 1.2])
plt.title('NER loss on the dev set')
plt.legend()
plt.savefig("ner_loss.png")
