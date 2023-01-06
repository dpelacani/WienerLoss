size = 256
path = os.path.abspath("/home/dp4018/data/ultrasound-data/Ultrasound-MRI-sagittal/")
train_transform = Compose([
                    Resize(size),
                    Lambda(lambda x: x / x.abs().max()),
                    Lambda(lambda x: clip_outliers(x, "outer")),
                    Lambda(lambda x: scale2range(x, [-1., 1.])),
                    # Normalize([0.09779735654592514], [0.16085614264011383])
                    ])

mask = create_mask((size,size), (0,3), (0,1))

ds = MaskedUltrasoundDataset2D(path, 
                                    mode="mri",
                                    transform=train_transform,
                                    mask=mask,
                                    maxsamples=1)
print(ds, "\n")
print(ds.info())

def make_model(nc=64):
    set_seed(42)
    channels = (16, 32, 64)#, 128, 256)
    model =  UNet(
    spatial_dims=2,
    in_channels=nc,
    out_channels=nc,
    channels=channels,
    strides=tuple([2 for i in range(len(channels))]), 
    num_res_units=1,
    act="mish")
    model = nn.DataParallel(model) 
    return model.to(device)
print(make_model())

msemodel = make_model(nc=x_sample.shape[0])
optimizer = torch.optim.Adam(msemodel.parameters(), lr=learning_rate)

mseloss     = nn.MSELoss(reduction="mean")

train_model(msemodel, optimizer, mseloss, train_loader,  valid_loader=valid_loader, nepochs=nepochs, log_frequency=150, sample_input=x_sample, sample_target=y_sample, device=device)


awmodel = make_model(nc=x_sample.shape[0])
optimizer = torch.optim.Adam(awmodel.parameters(), lr=learning_rate)

awloss     = AWLoss(filter_dim=2, method="fft", reduction="mean", store_filters="norm", epsilon=3e-15)

train_model(awmodel, optimizer, awloss, train_loader, valid_loader=valid_loader, nepochs=nepochs, log_frequency=150, sample_input=x_sample, sample_target=y_sample, device=device)