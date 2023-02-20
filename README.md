
# **MLOps for Everyone**

Being working as an MLops Engineer for the last two year, I think it is time to share out the knowledge and experience with everyone.
---

### **What is MLOps?**

MLops stands for "Machine Learning Operation". MLops is the amalgation of two fields: 
- DevOps 
- Data Science

So, when I say combination of two fields, you have to aware about the two domains. A good `MLops Engineer` comes from having these two Domain knowledge.
---

Now you may not understand how to write these lines of codes, but it is imperative to understand the purpose of this code.

```python
def preprocess_data(X, scaler=None):
    """Preprocess input data by standardise features 
    by removing the mean and scaling to unit variance"""
    if not scaler:
        scaler = StandardScaler()
        scaler.fit(X)
    X = scaler.transform(X)
    return X, scaler


def preprocess_labels(labels, encoder=None, categorical=True):
    """Encode labels with values among 0 and `n-classes-1`"""
    if not encoder:
        encoder = LabelEncoder()
        encoder.fit(labels)
    y = encoder.transform(labels).astype(np.int32)
    if categorical:
        y = np_utils.to_categorical(y)
    return y, encoder
```
---
The Purpose of this Blog is to wtite Daily Challenges by MLOps Engineer.



## Authors

- [@sidharth](https://www.github.com/sidharthkumarpradhan)


## 🚀 About Me
I'm a MLOps Engineer...


## License

[MIT](https://choosealicense.com/licenses/mit/)


## Support

For support, email support@learnmlops.ml

Follow www.learn-mlops.ml 


## 🛠 Skills
Javascript, HTML, CSS, Jquery, Data Science, Machine Learning, DevOps  ...

