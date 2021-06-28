from torch import nn
import torchvision as tv


def marker_classifier(num_class):
    model = tv.models.resnet34()
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
#         nn.Dropout(p=0.3),
        nn.Linear(num_ftrs, num_class-1),
        nn.Sigmoid()
    )
    return model