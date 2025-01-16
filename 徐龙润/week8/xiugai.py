#对load_data代码的修改如下：
    def random_train_sample(self):
        standard_question_index = list(self.knwb.keys())
        r1 = 1
        while r1<1:
            r1 = random.choice(standard_question_index)
        r2 = random.choice(standard_question_index)
        while r2==r1:
            r2 = random.choice(standard_question_index)
        a,p = random.sample(self.knwb[r1],2)
        n = random.sample(self.knwb[r2],1)[0]

        return [a,p,n]
#    对model.py的修改如下：
    def forward(self, sentence1, sentence2=None,sentence3=None, target=None):
        #同时传入两个句子
        if sentence2 is not None:
            vector1 = self.sentence_encoder(sentence1) #vec:(batch_size, hidden_size)
            vector2 = self.sentence_encoder(sentence2)
            vector3 = self.sentence_encoder(sentence3)
            #如果有标签，则计算loss
            if target is not None:
                return self.loss(vector1, vector2, target.squeeze())
            elif sentence3 is not None:
                return self.cosine_triplet_loss(vector1,vector2,vector3)
            #如果无标签，计算余弦距离
            else:
                return self.cosine_distance(vector1, vector2)
        #单独传入一个句子时，认为正在使用向量化能力
        else:
            return self.sentence_encoder(sentence1)



  #对main.py 的修改如下
def main(config):
    #创建保存模型的目录
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])
    #加载训练数据
    train_data = load_data(config["train_data_path"], config)
    #加载模型
    model = SiameseNetwork(config)
    # 标识是否使用gpu
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        logger.info("gpu可以使用，迁移模型至gpu")
        model = model.cuda()
    #加载优化器
    optimizer = choose_optimizer(config, model)
    #加载效果测试类
    evaluator = Evaluator(config, model, logger)
    #训练
    for epoch in range(config["epoch"]):
        epoch += 1
        model.train()
        logger.info("epoch %d begin" % epoch)
        train_loss = []
        for index, batch_data in enumerate(train_data):
            optimizer.zero_grad()
            if cuda_flag:
                batch_data = [d.cuda() for d in batch_data]
            input_id1, input_id2, input_id3 = batch_data   #输入变化时这里需要修改，比如多输入，多输出的情况
            loss = model(input_id1, input_id2, input_id3)
            train_loss.append(loss.item())
            # if index % int(len(train_data) / 2) == 0:
            #     logger.info("batch loss %f" % loss)
            loss.backward()
            optimizer.step()
        logger.info("epoch average loss: %f" % np.mean(train_loss))
        evaluator.eval(epoch)
    model_path = os.path.join(config["model_path"], "epoch_%d.pth" % epoch)
    torch.save(model.state_dict(), model_path)
    return
