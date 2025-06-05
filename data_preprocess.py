import pandas as pd


# length * N * feature_num
def data_process2(data):
    # multiindex_df = data.set_index(["order_book_id", "date"])
    multiindex_df = data
    data = data.reset_index()
    # print(multiindex_df)
    # stock_name = data['Stkcd'].unique()
    # for sn in stock_name:
    #     data['Stkcd'=sn]['Trddt']
    names = data["order_book_id"].unique()
    indexxx = pd.MultiIndex.from_product(
        [data["order_book_id"].unique(), data["date"].unique()],
        names=["order_book_id", "date"],
    )
    newdata = multiindex_df.reindex(indexxx)
    # print(len(newdata.index.levels[0]), len(newdata.index.levels[1]))
    newdata.reset_index()
    array_3d = newdata.to_numpy().reshape(
        len(newdata.index.levels[0]),
        len(newdata.index.levels[1]),
        len(newdata.columns),
    )  #
    # 确保 self.data[self.t, :, self.close_pos] 为return rate
    return array_3d.astype("float32").transpose(1, 0, 2), names.tolist()
