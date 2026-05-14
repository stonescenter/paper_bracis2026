import torch
from sklearn.metrics import roc_auc_score, average_precision_score
from torch_geometric.utils import negative_sampling
import numpy as np
import torch.nn.functional as F

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#data = d

def train(model, data, val_data, optimizer, criterion, n_epochs=100):

      model.train()
      optimizer.zero_grad()  # Clear gradients.
      #z = model.encode(train_data.x, train_data.edge_index)  # Perform a single forward pass.
      z = model.encode(data.x, data.edge_index)  # Perform a single forward pass.

      # We perform a new round of negative sampling for every training epoch:
      neg_edge_index = negative_sampling(
          edge_index=data.edge_index,
          num_nodes=data.num_nodes,
          num_neg_samples=data.edge_label_index.size(1),
          method='dense')

      edge_label_index = torch.cat(
          [data.edge_label_index, neg_edge_index],
          dim=-1,
      )

      edge_label = torch.cat([
          data.edge_label,
          data.edge_label.new_zeros(neg_edge_index.size(1))
      ], dim=0)


      out = model.decode(z, edge_label_index).view(-1)
      loss = criterion(out, edge_label)
      loss.backward()  # Derive gradients.
      optimizer.step()  # Update parameters based on gradients.
      return loss

def train_loader_(model, loader, optimizer, criterion):
      
      model.train()
      total_loss = 0
      total_loss = []

      for batch in loader: 
        batch.to(device)
        optimizer.zero_grad()  # Clear gradients.
        #z = model.encode(train_data.x, train_data.edge_index)  # Perform a single forward pass.
        z = model.encode(batch.x, batch.edge_index)  # Perform a single forward pass.

        # We perform a new round of negative sampling for every training epoch:
        neg_edge_index = negative_sampling(
            edge_index=batch.edge_index,
            num_nodes=batch.num_nodes,
            num_neg_samples=batch.edge_label_index.size(1),
            method='sparse')

        edge_label_index = torch.cat(
            [batch.edge_label_index, neg_edge_index],
            dim=-1,
        )

        edge_label = torch.cat([
            batch.edge_label,
            batch.edge_label.new_zeros(neg_edge_index.size(1))
        ], dim=0)


        out = model.decode(z, edge_label_index).view(-1)
        loss = criterion(out, edge_label)
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.

        #total_loss += loss.item()
        total_loss.append(loss.item())

      #return total_loss / len(loader.dataset)
      return np.mean(total_loss)

@torch.no_grad()
def evaluate_loader(model, loader):
    model.eval()
    y_pred, y_true = [], []
    
    for batch in loader:
        z = model.encode(batch.x, batch.edge_index)
        out = model.decode(z, batch.edge_label_index).view(-1).sigmoid()
        
        y_true.append(batch.edge_label.cpu().numpy())
        y_pred.append(out.cpu().numpy())
    
    
    y_pred, y_true = torch.cat(y_pred ), torch.cat(y_true)
    
    return roc_auc_score(y_true, y_pred)

@torch.no_grad()
def eval_link_predictor(model, data):
    model.eval()
    z = model.encode(data.x, data.edge_index)

    out = model.decode(z, data.edge_label_index).view(-1).sigmoid()
    return roc_auc_score(data.edge_label.cpu().numpy(), out.cpu().numpy())


def train_link_predictor(
    model, train_data, test_data, val_data, optimizer, criterion, n_epochs=100
):
    best_val_auc = final_test_auc = 0
    model.train()
    val_auc_avg = []
    for epoch in range(1, n_epochs + 1):

       
        optimizer.zero_grad()
        #z = model.encode(train_data.x, train_data.edge_index)
        z = model.encode(train_data.x, train_data.edge_index)


        # sampling training negatives for every training epoch
        neg_edge_index = negative_sampling(
            edge_index=train_data.edge_index, num_nodes=train_data.num_nodes,
            num_neg_samples=train_data.edge_label_index.size(1), method='sparse')

        edge_label_index = torch.cat(
            [train_data.edge_label_index, 
             neg_edge_index],
            dim=-1,
        )
        edge_label = torch.cat([
            train_data.edge_label,
            train_data.edge_label.new_zeros(neg_edge_index.size(1))
        ], dim=0)

        # prediction of model
        out = model.decode(z, edge_label_index).view(-1)
        loss = criterion(out, edge_label)
        loss.backward()
        optimizer.step()

        val_auc = eval_link_predictor(model, val_data)
        test_auc = eval_link_predictor(model, test_data)

        val_auc_avg.append(val_auc)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            final_test_auc = test_auc

        if epoch % 10 == 0:
            print(f"Epoch: {epoch:03d}, Train Loss: {loss:.3f}, Test AUC: {test_auc:.3f}, Val AUC: {val_auc:.3f}, ")

    print(f'Final Test: {final_test_auc:.3f}')
    print(f'Final Val: {best_val_auc:.3f}')
    print(f'Final Val AUC avg:{ np.mean(val_auc_avg)}')
    return model




def train_split(encoder, predictor,
                train_data, val_data, optimizer, criterion, n_epochs=100):
    
    encoder.train()
    predictor.train()
    optimizer.zero_grad()

    # Get node embeddings from the encoder using the training graph's edges
    z = encoder(train_data.x, train_data.edge_index)

    # Perform negative sampling on the fly for training
    neg_edge_index = negative_sampling(
        edge_index=train_data.edge_index, num_nodes=train_data.num_nodes,
        num_neg_samples=train_data.edge_label_index.size(1), method='sparse')
    
    edge_label_index = torch.cat([train_data.edge_label_index, neg_edge_index], dim=-1)
    edge_label = torch.cat([torch.ones(train_data.edge_label_index.size(0)),
                            torch.zeros(neg_edge_index.size(0))], dim=0)

    # Get link predictions
    out = predictor(z, edge_label_index).view(-1)
    # Calculate loss
    loss = criterion(out, edge_label)
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def test_split(encoder, predictor, data):
    encoder.eval()
    predictor.eval()
    z = encoder(data.x, data.edge_index) # Use data's edge_index for message passing
    out = predictor(data.edge_label_index, z).view(-1)
    
    # Calculate evaluation metrics (AUC and AP)
    auc_score = roc_auc_score(data.edge_label.cpu().numpy(), out.sigmoid().cpu().numpy())
    ap_score = average_precision_score(data.edge_label.cpu().numpy(), out.sigmoid().cpu().numpy())
    return auc_score, ap_score

# 
# If an edge exists → use real features
# If negative → use zeros (common baseline)
#
def get_edge_features(full_edge_index, full_edge_attr, query_edge_index):
    edge_dict = {
        (u.item(), v.item()): full_edge_attr[i]
        for i, (u, v) in enumerate(full_edge_index.t())
    }

    features = []
    for u, v in query_edge_index.t():
        feat = edge_dict.get((u.item(), v.item()))
        if feat is None:
            feat = torch.zeros(full_edge_attr.size(1), device=full_edge_index.device)
        features.append(feat)

    return torch.stack(features)

def get_batch_edge_attr(batch):
    edge_dict = {
        (u.item(), v.item()): batch.edge_attr[i]
        for i, (u, v) in enumerate(batch.edge_index.t())
    }

    feats = []
    for u, v in batch.edge_label_index.t():
        feats.append(edge_dict.get((u.item(), v.item()),
                                   torch.zeros(batch.edge_attr.size(1))))
    return torch.stack(feats)

def get_batch_edge_attr(batch):
    device = batch.edge_index.device
    edge_dim = batch.edge_attr.size(1)

    edge_dict = {
        (u.item(), v.item()): batch.edge_attr[i]
        for i, (u, v) in enumerate(batch.edge_index.t())
    }

    feats = []
    for u, v in batch.edge_label_index.t():
        feat = edge_dict.get(
            (u.item(), v.item()),
            torch.zeros(edge_dim, device=device)  # ← FIX
        )
        feats.append(feat)

    return torch.stack(feats)

def train_sage(model, predictor, data, optimizer, criterion):
    model.train()
    predictor.train()
    optimizer.zero_grad()

    z = model(data.x.to(device),
                data.edge_index.to(device))

    edge_index = data.edge_label_index.to(device)
    labels = data.edge_label.to(device).float()

    edge_attr = get_edge_features(
        data.edge_index,
        data.edge_attr,
        data.edge_label_index
    ).to(device)

    logits = predictor(z, edge_index, edge_attr)

    #loss = F.binary_cross_entropy_with_logits(logits, labels)
    loss = criterion(logits, labels)
    loss.backward()
    optimizer.step()

    return loss.item()

@torch.no_grad()
def evaluate_sage(model, predictor, data):
    model.eval()
    predictor.eval()

    z = model(data.x.to(device),
                data.edge_index.to(device))

    edge_attr = get_edge_features(
        data.edge_index,
        data.edge_attr,
        data.edge_label_index
    ).to(device)

    logits = predictor(z,
                       data.edge_label_index.to(device),
                       edge_attr)

    pred = torch.sigmoid(logits).cpu()
    return roc_auc_score(data.edge_label.cpu(), pred)

def train_sage_loader(model, predictor, loader, optimizer, criterion):
    model.train()
    predictor.train()

    total_loss = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        z = model(batch.x.to(device), batch.edge_index.to(device))

        edge_attr = get_batch_edge_attr(batch).to(device)

        logits = predictor(z,
                           batch.edge_label_index,
                           edge_attr)

        loss = criterion(logits, batch.edge_label.to(device).float())
        #loss = F.binary_cross_entropy_with_logits(logits, batch.edge_label.to(device).float())

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch.num_edges

    return total_loss / len(loader.dataset)

@torch.no_grad()
def evaluate_sage_loader(model, predictor, loader):
    model.eval()
    predictor.eval()

    all_preds = []
    all_labels = []

    for batch in loader:
        batch = batch.to(device)
        z = model(batch.x, batch.edge_index)

        edge_attr = get_batch_edge_attr(batch).to(device)

        preds = predictor(z,
                           batch.edge_label_index,
                           edge_attr).sigmoid().cpu()

    
        #preds = torch.sigmoid(logits).view(-1).cpu()
        #labels = batch.edge_label.view(-1).cpu()
        #print(batch.edge_label.shape)
        #print(batch.edge_label.unique())
        #preds = torch.sigmoid(logits).detach().cpu()
        labels = batch.edge_label.cpu()


        # --- FORCE CORRECT SHAPES ---
        preds = preds.reshape(-1)
        labels = labels.reshape(-1)

        # --- FORCE BINARY TYPE ---
        labels = labels.to(torch.int64)
        #print(batch.edge_label.shape)
        #print(batch.edge_label.unique())

        all_preds.append(preds)
        all_labels.append(labels)

        #print("pred shape:", preds.shape)
        #print("label shape:", labels.shape)
        #print("label values unique:", labels.unique())
        #print("label values:", labels)
        

    y_pred = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_labels).numpy()

    #print(f" y_pred: {y_pred}")
    #print(f" y_true: {y_true}")
    # Skip AUC if only one class present
    if len(set(y_true.tolist())) < 2:
        return float("nan")
    
    return roc_auc_score(y_true, y_pred)

@torch.no_grad()
def evaluate_gat(model, predictor, data):
    model.eval()
    predictor.eval()

    data = data.to(device)

    z = model(
        data.x,
        data.edge_index
    )
    
    edge_attr = get_edge_features(
        data.edge_index,
        data.edge_attr,
        data.edge_label_index
    )

    logits = predictor(
        z,
        data.edge_label_index,
        edge_attr
    )

    preds = torch.sigmoid(logits).view(-1).cpu()
    labels = data.edge_label.view(-1).cpu()

    # Avoid sklearn crash
    if len(labels.unique()) < 2:
        return float("nan")

    return roc_auc_score(labels.numpy(), preds.numpy())

def train_gat(model, predictor, train_data, test_data, val_data, optimizer, criterion, n_epochs=100):

    best_val_auc = final_test_auc = 0 
    train_data = train_data.to(device)
    
    for epoch in range(1, n_epochs + 1):
        model.train()
        predictor.train()
        optimizer.zero_grad()
        
        z = model(
            train_data.x,
            train_data.edge_index
        )

        # the MLP relies on magnitude + direction
        # normalize() forces all embeddings to lie on the unit hypersphere
        #z = F.normalize(z, dim=-1) 
        
        edge_attr = get_edge_features(
            train_data.edge_index,
            train_data.edge_attr,
            train_data.edge_label_index
        )

        logits = predictor(
            z,
            train_data.edge_label_index,
            edge_attr
        )

        labels = train_data.edge_label.float()

        #loss = F.binary_cross_entropy_with_logits(logits, labels)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        loss = loss.item()
        
        val_auc = evaluate_gat(model, predictor, val_data)
        test_auc = evaluate_gat(model, predictor, test_data)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            final_test_auc = test_auc

        if epoch % 10 == 0:
            print(f"Epoch: {epoch:03d}, Train Loss: {loss:.3f}, Val AUC: {val_auc:.3f}, Test AUC: {test_auc:.3f}")

    print(f'Final Test: {final_test_auc:.3f}')

    return model


