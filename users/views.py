from django.shortcuts import render, HttpResponse
from django.contrib import messages
from .forms import UserRegistrationForm
from .models import UserRegistrationModel
from django.conf import settings
import os


# Create your views here.
def UserRegisterActions(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            print('Data is Valid')
            form.save()
            messages.success(request, 'You have been successfully registered')
            form = UserRegistrationForm()
            return render(request, 'UserRegistrations.html', {'form': form})
        else:
            messages.success(request, 'Email or Mobile Already Existed')
            print("Invalid form")
    else:
        form = UserRegistrationForm()
    return render(request, 'UserRegistrations.html', {'form': form})


def UserLoginCheck(request):
    if request.method == "POST":
        loginid = request.POST.get('loginname')
        pswd = request.POST.get('pswd')
        print("Login ID = ", loginid, ' Password = ', pswd)
        try:
            check = UserRegistrationModel.objects.get(loginid=loginid, password=pswd)
            status = check.status
            print('Status is = ', status)
            if status == "activated":
                request.session['id'] = check.id
                request.session['loggeduser'] = check.name
                request.session['loginid'] = loginid
                request.session['email'] = check.email
                print("User id At", check.id, status)
                return render(request, 'users/UserHome.html', {})
            else:
                messages.success(request, 'Your Account Not at activated')
                return render(request, 'UserLogin.html')
        except Exception as e:
            print('Exception is ', str(e))
            pass
        messages.success(request, 'Invalid Login id and password')
    return render(request, 'UserLogin.html', {})


def UserHome(request):
    return render(request, 'users/UserHome.html', {})


def ViewPaySimData(request):
    path = os.path.join(settings.MEDIA_ROOT, 'PaySimDataset.csv')
    import pandas as pd
    df = pd.read_csv(path, nrows=101)
    df = df.to_html(index=None)
    return render(request, 'users/viewPaySimData.html', {'data': df})


def edaAnalysis(request):
    # from .utility.edaprocess import start_eda_process
    # obj = start_eda_process()
    return render(request, 'users/paysimeda.html', {})


def userMachineLearning(request):
    from .utility.paysimmlcode import start_ml_models
    rf, lgbm, xgbr, logreg, dt, knn, svm = start_ml_models()
    return render(request, 'users/paysimresults.html', {'rf': rf,'lgbm': lgbm, 'xgbr': xgbr, 'logreg': logreg, 'dt': dt, 'knn': knn, 'svm': svm})


def testPredictions(request):
    if request.method=='POST':
        amount = float(request.POST.get('amount'))
        sender_old_balance = float(request.POST.get('sender_old_balance'))
        sender_new_balance = float(request.POST.get('sender_new_balance'))
        receiver_old_balance = float(request.POST.get('receiver_old_balance'))
        receiver_new_balance = float(request.POST.get('receiver_new_balance'))
        type_CASH_OUT = float(request.POST.get('type_CASH_OUT'))
        type_DEBIT = float(request.POST.get('type_DEBIT'))
        type_PAYMENT = float(request.POST.get('type_PAYMENT'))
        type_TRANSFER = float(request.POST.get('type_TRANSFER'))
        type2_CM = float(request.POST.get('type2_CM'))
        testSet = [amount,sender_old_balance,sender_new_balance,receiver_old_balance,receiver_new_balance,type_CASH_OUT,
                   type_DEBIT,type_PAYMENT,type_TRANSFER,type2_CM]
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        test = sc.fit_transform([testSet])
        model_file = os.path.join(settings.MEDIA_ROOT, 'pysimmodel.alex')
        import pickle
        model = pickle.load(open(model_file, 'rb'))
        # test = test.reshape(-1, 1)
        import numpy as np
        pred = model.predict([testSet])
        if pred[0]==0:
           msg = 'Fraud transaction'
        else:
           msg = 'Not Fraud'
        return render(request, 'users/predictionform.html', {'pred': msg})
    else:
        return render(request, 'users/predictionform.html', {})
