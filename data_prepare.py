import gzip
import json
import time
import os
import random
  
meta_files = ["./raw_data/meta_Books.json.gz",
             "./raw_data/meta_Electronics.json.gz",
             "./raw_data/meta_Clothing_Shoes_and_Jewelry.json.gz",]

def get_meta_data(path):
    count = 0
    file_name = path.split("/")[-1].split(".")[0]
    line = int(os.popen("wc -l " + path).read().split(" ")[0])
    meta_item = {}
    cate_index = {}
    for review in parse(path):
        count +=1
        asin = review["asin"]
        if review['category'] == []:
            continue
        cate = review['category'][-1]
        meta_item[asin] = cate
        if(cate not in cate_index):
            cate_index[cate] = len(cate_index)
        if count % 1000 == 0:
            print("file: %s step: %d/%d" % (file_name, count, line), end = "\r")
    return meta_item, cate_index

#解析数据
def build_data(path, meta_data):
    count = 0
    file_name = path.split("/")[-1].split(".")[0]
    line = int(os.popen("wc -l " + path).read().split(" ")[0])
    sample = {}
    item_index = {}
    user_index = {}
    asin_not_found = 0
    for review in parse(path):
        count +=1
        timestamp = int(review['unixReviewTime'])
        reviewerID = review['reviewerID']
        asin = review["asin"]
        score = review['overall']
        if asin in meta_data:
            cate = meta_data[asin]
            if(reviewerID not in sample):
                sample[reviewerID] = [[asin,timestamp,score, cate]]
                user_index[reviewerID] = len(user_index)
            else:
                sample[reviewerID] += [[asin,timestamp,score, cate]]
            if(asin not in item_index):
                item_index[asin] = len(item_index)
        else:
            asin_not_found += 1
        if count % 1000 == 0:
            print("file: %s step: %d/%d  asin not found = %d" % (file_name, count, line, asin_not_found), end = "\r")
    return sample,user_index,item_index

import random
def get_week(timestamp):
    time_obj = time.localtime(timestamp)
    return int(time.strftime("%w", time_obj))+1
def get_hour(timestamp):
    time_obj = time.localtime(timestamp)
    return int(time.strftime("%H", time_obj))
def get_month(timestamp):
    time_obj = time.localtime(timestamp)
    return int(time.strftime("%m", time_obj))
def get_diff_hours(t1,t2):
    return int((t2-t1)/60/60)
def get_diff_days(t1,t2):
    return int((t2-t1)/60/60/24)
def get_diff_months(t1,t2):
    return int((t2-t1)/60/60/24/30)
def get_sample(samples, users, items, name, index, cate_index, meta_item):
    flag = 0
    file_name = ["2017%02d" %i for i in range(1,13)]
    #file_name += ["2018%02d" %i for i in range(1,10)]
    file_list = []
    for i in file_name:
        file_list += [open("./result_data/"+name+"/part-"+i,"w")]
    line = int(os.popen("wc -l " + data_files[index]).read().split(" ")[0])
    asin_not_found = 0
    for review in parse(data_files[index]):
        flag +=1
        timestamp = int(review['unixReviewTime'])
        reviewerID = review['reviewerID']
        asin = review["asin"]
        score = review['overall']
        if asin in meta_item:
            seq = samples[reviewerID]
            cate = meta_item[asin]
            seq = [x for x in seq if x[1] < timestamp]#筛选靠前的record
            if(len(seq) == 0): continue

            seq = sorted(seq,key = lambda x:x[1])[-50:]
            clicks = [x[0] for x in seq]
            timestamps = [x[1] for x in seq]
            scores = [x[2] for x in seq]
            cates = [x[3] for x in seq]
            target_score = score
            #print(target_score)
            target_timestamp = timestamp
            #print(target_timestamp)
            target_item = asin
            #print(items[target_item])
            date = time.strftime("%Y%m",time.localtime(target_timestamp))
            if date not in file_name:
                continue

            click_seq = ",".join(["0"]*(50-len(clicks))+[str(items[x]+1) for x in clicks])
            #print(click_seq)
            cate_seq = ",".join(["0"]*(50-len(clicks))+[str(cate_index[x]+1) for x in cates])
            #print(click_seq)
            convert_mask = ",".join(["0"]*(50-len(clicks))+[str(int(int(x)==5)) for x in scores])
            #print(convert_mask)
            weeks = ",".join(["0"]*(50-len(clicks))+[str(get_week(x)) for x in timestamps])
            #print(weeks)
            months = ",".join(["0"]*(50-len(clicks))+[str(get_month(x)) for x in timestamps])
            #print(hours)
            diff_months = ",".join(["0"]*(50-len(clicks))+[str(get_diff_months(x, target_timestamp)) for x in timestamps])
            #print(diff_hours)
            diff_days = ",".join(["0"]*(50-len(clicks))+[str(get_diff_days(x, target_timestamp)) for x in timestamps])
            #print(diff_days)

            target_week = str(get_week(target_timestamp))
            target_hour = str(get_hour(target_timestamp))
            target_month = str(get_month(target_timestamp))
            target_user = str(users[reviewerID]+1)
            target_cate = str(cate_index[cate]+1)
            #print(target_week,target_hour,target_user)
            label = "1"
            target_item = items[target_item]+1
            res = label+"\t"+str(target_user)+"\t"+str(target_item)+"\t"+str(target_cate)+"\t"\
                 +target_week+"\t"+target_month+"\t"+click_seq+"\t"+cate_seq+"\t"+convert_mask+"\t"\
                 +weeks+"\t"+months+"\t"+diff_months+"\t"+diff_days + "\t" + str(target_timestamp)
            file_list[file_name.index(date)].write(res + "\n")
        else:
            asin_not_found += 1
        if flag % 100 == 0:
            print("file: %s step: %d/%d asin not found = %d" % (name, flag, line, asin_not_found), end = "\r")    
    for f in file_list:
        f.close()

def get_aux_sample(name):
    path_list = []
    for i in os.listdir("./result_data/"+name):
        path_list += ["./result_data/"+name+"/" + i]
    
    items = set()
    for path in path_list:
        with open(path,"r") as f:
            for line in f:
                item = line.split("\t")[2]
                cate = line.split("\t")[3]
                items.add(item + "\t" + cate)
    items = list(items)  
    for path in path_list:            
        with open("./result_data/aux_"+name+"/"+path.split("/")[-1],"w") as w:
            with open(path,"r") as f:
                for line in f:
                    item = line.split("\t")[2]
                    cate = line.split("\t")[3]
                    item_ = [item + "\t" + cate]
                    sample = line
                    for i in range(1):#负采样比例
                        neg = random.randint(0,len(items)-1)
                        while(items[neg] in item_):
                            neg = random.randint(0,len(items)-1)
                        item_ += [items[neg]]
                        tmp = line.strip().split("\t")
                        tmp[0] = "0"
                        tmp[2] = items[neg].split("\t")[0]
                        tmp[3] = items[neg].split("\t")[1]
                        pos_items = tmp[6].split(",")
                        neg_items = []
                        neg_cates = []
                        for i in range(100):#序列的负采样
                            neg = random.randint(0,len(items)-1)
                            while(items[neg].split("\t")[0] in pos_items):
                                neg = random.randint(0,len(items)-1)
                            neg_items += [items[neg].split("\t")[0]]
                            neg_cates += [items[neg].split("\t")[1]]
                        w.write(line.strip() + "\t" + ",".join(neg_items[:50]) + "\t" + ",".join(neg_cates[:50]) + "\n")
                        w.write("\t".join(tmp) + "\t" + ",".join(neg_items[50:]) + "\t" + ",".join(neg_cates[50:]) + "\n")
    print(len(items))

    
if main == "__main__":
    print("step 1: \n")
    book_meta, book_cate_index = get_meta_data(meta_files[0])
    print("step 2: \n")
    book_sample,book_user,book_item = build_data(data_files[0], book_meta)
    print("step 3: \n")
    get_sample(book_sample,book_user,book_item,"book",0, book_cate_index, book_meta)
    print("step 4: \n")
    get_neg_sample("book")
    print(len(book_cate_index), len(book_user), len(book_item))
    get_aux_sample("book")